#pragma once
#include "model_loader.hpp"
#include "encoder.hpp"
#include "mel.hpp"
#include "prediction.hpp"
#include "joint.hpp"
#include "rnnt.hpp"
#include "prompt_kernel.hpp"
#include "decode_types.hpp"
#include "transcription.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace pk {

// One step of the buffered schedule. Sample positions are absolute from the
// start of the stream.
struct BufferedStepPlan {
    int64_t edge = 0;          // right edge of the step's chunk
    int64_t num = 0;           // samples added by this step (edge - previous edge)
    bool is_last = false;      // NeMo's is_last_chunk: this step reaches the end of the audio
    int64_t left = 0;          // buffer context after the add, in samples (NeMo ContextSize)
    int64_t chunk = 0;
    int64_t right = 0;
    int64_t win_start = 0;     // window = audio[win_start, edge), length left + chunk + right
};

// The steps of NeMo's buffered streaming loop (speech_to_text_streaming_infer_rnnt.py,
// batch 1) as a pure schedule with no model. A step whose right edge is e is
// emitted only once more than e samples have arrived, so it is the last step only
// when nothing follows it. Feeding the audio in any block sizes therefore gives the
// same steps. The context arithmetic is a port of NeMo's
// ContextSize.add_frames_get_removed_ (streaming_utils.py), in samples.
class BufferedSchedule {
public:
    // left, chunk, right: frames (80 ms = 1280 samples).
    BufferedSchedule(int left, int chunk, int right);

    // Appends every step whose edge is strictly inside the `received` samples.
    void advance(int64_t received, std::vector<BufferedStepPlan>& out);

    // Once the total length is known (end of stream): the last step, if audio
    // remains after the previous step. Returns false when no step is left.
    bool finish(int64_t total, BufferedStepPlan& out);

    bool done() const { return done_; }

private:
    BufferedStepPlan make(int64_t edge, bool is_last);
    void add_frames(int64_t num, bool is_last);

    int64_t exp_left_, exp_chunk_, exp_right_;   // expected context, samples
    int64_t ctx_left_ = 0, ctx_chunk_ = 0, ctx_right_ = 0;
    int64_t left_edge_ = 0;                      // previous step's right edge
    int64_t next_edge_ = 0;                      // edge of the next step, if it is not the last
    bool done_ = false;
};

// Buffered (re-encoding) streaming for the unified checkpoints
// (att_context_style chunked_limited_with_rc, e.g. nvidia/nemotron-labs-unified-asr-0.6b).
//
// Each step re-runs the whole model on the raw audio window [left | chunk | right]:
// the log-mel is recomputed and per-feature normalized over the window, the
// encoder runs with full attention under the window-relative RC mask (RcMask),
// and the language prompt is applied. Only the chunk frames are decoded, and the
// RNN-T decoder state is carried between steps. Committed tokens never change.
// This mirrors NeMo's speech_to_text_streaming_infer_rnnt.py (B = 1). The steps
// come from BufferedSchedule.
//
// left, chunk and right are in 80 ms encoder frames (1280 samples at 16 kHz).
class BufferedStreamingSession {
public:
    // target_lang selects the language prompt (empty = the model default). An
    // unknown locale for a prompt model throws std::runtime_error.
    BufferedStreamingSession(const ModelLoader& ml, int left, int chunk, int right,
                             const std::string& target_lang = "");

    // Feed 16 kHz mono PCM. Runs every step that is ready. Throws after finalize().
    void feed(const float* pcm, int n);

    // End of stream: runs the last step on the remaining audio, if any was received.
    // Calling it again does nothing.
    void finalize();

    // Start a new utterance: clears the audio, the context, the decoder state and the text.
    void reset();

    // Newly committed text since the last call (empty if none).
    std::string take_new_text();

    // Words finalized since the last drain. A word is final once the next word starts,
    // and every word is final after finalize().
    std::vector<Word> drain_words();

    // All committed token ids so far.
    const std::vector<int32_t>& tokens() const { return hyp_; }
    // Committed transcript (all tokens detokenized).
    const std::string& text() const { return text_; }

    // Per-step trace, recorded only when enabled (tests and diagnostics).
    struct Step {
        BufferedStepPlan plan;         // schedule of the step
        int enc_frames = 0;            // encoder output frames for the window (NeMo's enc_len)
        int dec_frames = 0;            // frames decoded in this step (NeMo's dec_len)
        std::vector<int32_t> tokens;   // token ids emitted by this step
        std::vector<float> enc_tm;     // decoded frames after the prompt, time-major [dec_frames, d_model]
    };
    void set_trace(bool on) { trace_on_ = on; }
    const std::vector<Step>& trace() const { return trace_; }

    int left_frames() const { return left_; }
    int chunk_frames() const { return chunk_; }
    int right_frames() const { return right_; }
    int64_t total_received() const { return received_; }
    bool finalized() const { return finalized_; }
    // Encoder frame stride in seconds (80 ms for this model).
    double frame_sec() const { return frame_sec_; }

private:
    // Run one step of the schedule.
    void run_step(const BufferedStepPlan& plan);
    void regroup_words(bool flush_all);

    const ModelLoader& ml_;
    int left_, chunk_, right_;          // frames
    int d_model_ = 0;
    int blank_id_ = 0;
    int max_symbols_ = 10;
    int prompt_index_ = -1;

    MelFrontend mel_;
    Encoder enc_;
    PredictionNet pred_;
    Joint joint_;
    PromptKernel prompt_;
    RnntDecodeState state_;
    BufferedSchedule schedule_;

    // Audio. pcm_[k] is the sample at absolute index origin_ + k.
    std::vector<float> pcm_;
    int64_t origin_ = 0;
    int64_t received_ = 0;
    int64_t frames_done_ = 0;           // encoder frames decoded so far (absolute offset)
    bool finalized_ = false;

    // Text and words, as StreamingSession.
    std::vector<int32_t> hyp_;
    std::string text_;
    size_t text_taken_ = 0;
    std::vector<TokenInfo> word_tokens_;
    std::vector<Word> words_;
    size_t words_finalized_ = 0;
    size_t words_taken_ = 0;
    float frame_sec_f_ = 0.0f;
    double frame_sec_ = 0.0;

    bool trace_on_ = false;
    std::vector<Step> trace_;
};

} // namespace pk
