# VoiceBank + DEMAND dataset

Source: Cassia Valentini-Botinhao, University of Edinburgh (2017),
[Noisy speech database for training speech enhancement algorithms and TTS models](https://doi.org/10.7488/ds/2117).
The original recordings are 48 kHz. This project resamples them to 16 kHz mono.

The dataset is available under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
The [publisher's license](https://datashare.ed.ac.uk/bitstreams/7e7bc32f-e94a-436a-8e97-416a972c7e1a/download)
applies to the bundled excerpts, transcripts and derived audio as well.
Bundled demo recordings are resampled excerpts. Enhanced backup recordings are modified by our U-Net.

## What GitHub includes

- `sources.json`: official archive URLs, expected counts and locally verified audio-archive SHA-256 checksums.
- `processed_manifest.csv.gz`: filenames, sizes and SHA-256 checksums of all 24,792 processed benchmark WAV files.
- `demo/`: six noisy demonstration recordings at 16 kHz with their reference transcripts.
- `dev_correction_set.json`: the 60 synthetic examples used for correction tuning.
- `../scripts/download_dataset.py`: a downloader and extractor for the full audio dataset and official test transcripts.

The full dataset is restored from the original publisher instead of duplicating multi-gigabyte archives in Git history.
The local working copy occupies about 14 GB because it contains downloaded ZIPs, 48 kHz audio and 16 kHz audio.
The full 28-speaker train archives plus test archives download approximately 5.6 GB in decimal units.

## Quick demonstration without the full dataset

From the repository root, run `python scripts/setup_demo.py --download-models`.
It copies the six bundled samples and the trained checkpoint into the paths expected by the app.
Run this online once before using the offline launcher.

## Restore the benchmark

```bash
# Test set only (824 clean/noisy pairs and reference transcripts):
python scripts/download_dataset.py --split test --resample

# Full 28-speaker training pool and test set:
python scripts/download_dataset.py --split all --resample

# Check pairing, sample rates and clipping:
python -m src.preprocessing.validate_dataset --all
```

Archives are saved under `data/raw/voicebank_demand/`. Extracted directories have the same names as
the archives without `.zip`. The processed recordings are under `data/processed/voicebank_demand_16k/`.
The downloader preserves existing matching-size extracted files and verifies archive checksums where provided.
Rebuilding with a different audio-library version can change processed sample checksums slightly.

## Exact split used in the experiment

| Partition | Recordings | Speakers |
|---|---:|---|
| Training | 10,700 | 26 speakers from the 28-speaker training pool |
| Validation | 872 | p286 and p287 |
| Test | 824 | p232 and p257 |

The 28-speaker pool has 11,572 recordings. The enhancer uses speaker-disjoint partitions.
Correction safeguards were also informed by earlier test error analysis, so correction development was not fully blind.
