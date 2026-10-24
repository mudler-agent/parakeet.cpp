### phonon-2.5 (Fermion Research)

Source: [FermionResearch/Phonon-2.5](https://huggingface.co/FermionResearch/Phonon-2.5) (25 languages) and [FermionResearch/Phonon-2.5-English](https://huggingface.co/FermionResearch/Phonon-2.5-English) (English only) by [Fermion Research](https://huggingface.co/FermionResearch), derived from NVIDIA's [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) · TDT transducer (FastConformer) · heads: TDT · license: [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)

| File | Variant | Size | Runs on |
|---|---|---:|---|
| `phonon-2.5-f16.gguf` | F16 | 1441.0 MB | any backend |
| `phonon-2.5-q8_0.gguf` | Q8_0 | 940.7 MB | any backend |
| `phonon-2.5-q4_k.gguf` | Q4_K | 675.2 MB | any backend |
| `phonon-2.5-packed.gguf` | packed five-value encoder | 241.7 MB | CPU only, offline only |
| `phonon-2.5-english-f16.gguf` | F16, English model | 1441.0 MB | any backend |
| `phonon-2.5-english-q8_0.gguf` | Q8_0, English model | 940.7 MB | any backend |
| `phonon-2.5-english-q4_k.gguf` | Q4_K, English model | 675.2 MB | any backend |
| `phonon-2.5-english-packed.gguf` | packed five-value encoder, English model | 241.6 MB | CPU only, offline only |

> There is no NeMo reference for these models, so there is no WER-vs-NeMo column. The F32 conversion was compared with the stock transformers `ParakeetForTDT` in fp32: identical token ids on 25 clips and an encoder difference of at most 1.0e-6 on identical input. Our own WER on 180 FLEURS utterances (the first 30 of each of English, German, French, Italian, Spanish and Polish; 4,204 words; plain normalizer): F16 5.42%, Q8_0 5.47%, Q4_K 5.71%, packed 5.47%, against 4.57% for NVIDIA's v3 F16 on the same utterances (Phonon-2.5 is worse on this subset, by 0.86 points, interval 0.12 to 1.60). That is a small sample and not comparable with the figures on the upstream cards; see [phonon.md](https://github.com/mudler/parakeet.cpp/blob/master/docs/phonon.md) for the tables, the intervals and the English model.

- **F16, Q8_0, Q4_K** hold ordinary weights, expanded from Fermion's compressed container, so they load like any other v3 file and run on every backend.
- **Packed** keeps the 264 encoder linear layers in Fermion's five-value form (each weight is 0, plus or minus a per-row `lo`, or plus or minus a per-row `hi`) and runs a native CPU kernel on two ternary planes. It does not run on GPU backends or in streaming mode, and parakeet.cpp refuses to load it there. For GPU use, take an F16 or Q8_0 file.
- **Changes:** these files are converted here, not trained. Nothing was trained or fine-tuned by the parakeet.cpp project. The models were trained by NVIDIA (the base) and Fermion Research (Phonon-2.5). The upstream notice is in `NOTICE-phonon-2.5.txt`; the training audio of Phonon-2.5 includes CHiME-6 (CC-BY-SA-4.0) and SPGISpeech (Kensho terms).

```bash
huggingface-cli download mudler/parakeet-cpp-gguf phonon-2.5-q8_0.gguf --local-dir models/
build/examples/cli/parakeet-cli transcribe --model models/phonon-2.5-q8_0.gguf --input audio.wav
```
