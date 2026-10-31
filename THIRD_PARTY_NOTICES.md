# Third-party sources

## Dataset and demonstration recordings

Cassia Valentini-Botinhao (2017), *Noisy speech database for training speech enhancement algorithms and TTS models*,
University of Edinburgh, https://doi.org/10.7488/ds/2117.
License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
See [data/README.md](data/README.md). Demo excerpts were resampled to 16 kHz mono.
Enhanced audio in the saved replay was additionally processed by the project's U-Net.
The publisher describes the source speech as VCTK/Voice Bank and environmental noise as including DEMAND.

## Pretrained components

- [OpenAI Whisper](https://github.com/openai/whisper): pretrained `base.en`, used for inference.
- [GPT-2](https://huggingface.co/openai-community/gpt2): pretrained sentence scorer, used locally for inference.
- [CMU Pronouncing Dictionary](https://github.com/cmusphinx/cmudict): pronunciation data through `cmudict`.

These components retain their upstream licenses and attribution. Their downloaded model caches are not copied into this repository.
The repository includes its own trained U-Net checkpoint, rather than claiming to have trained Whisper or GPT-2.

## Presentation fonts

Epilogue and Fraunces are distributed under the SIL Open Font License 1.1.
See `presentation/assets/fonts/OFL-Epilogue.txt` and `OFL-Fraunces.txt` for their original notices.
The font files are supplied to preserve the presentation's appearance.

## Code rights

The project author has not selected a blanket software license for the original project code.
Publishing the repository for academic review does not change the licenses of third-party materials.
