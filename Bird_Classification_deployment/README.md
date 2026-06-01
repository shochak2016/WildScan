# WildScan Bird-Sound Classifier

Classifies a bird recording into one of **50 common North American bird genera**
from audio alone. Trains on a GPU; the shipped model runs on **CPU** in ~0.2–0.4 s
per recording.

Genus-level (not species) by design — it matches the WildScan plant model's
`taxon_genus_name` convention and collapses acoustically near-identical species
(e.g. the *Empidonax* flycatchers, the *Melospiza* sparrows) into one correct
answer, which is both more honest and more accurate.

---

## Pipeline at a glance

The **entire pipeline lives in one file**, `bird_classifier.py`, exposed as subcommands:

```
Xeno-canto API ──► download ──► clean ──► window+select ──► PCEN spectrogram ──► train (GPU) ──► CPU inference
 (50 genera)      (mp3s)      (filter)   (band-SNR)        (.npy)              (sweep+ensemble)  (predict)
```

| Stage | Command |
|-------|---------|
| Download | `bird_classifier.py download` |
| Clean → spectrogram | `bird_classifier.py preprocess` |
| Train (single model) | `bird_classifier.py train` |
| Hyperparameter sweep | `bird_classifier.py sweep` |
| Ensemble build | `bird_classifier.py ensemble` |
| TTA evaluation | `bird_classifier.py tta-eval` |
| **CPU inference** | `bird_classifier.py predict <audio>` |
| Genus list (config) | `genus_list.json` |
| Dataset EDA | `eda.ipynb` |

---

## 1. Dataset — Xeno-canto

Source: the [Xeno-canto](https://xeno-canto.org) API **v3** (v2 is retired; v3
requires a free API key in `XC_API_KEY` or a gitignored `.xc_api_key` file).

**Selection criteria** (`bird_classifier.py download`):
- **50 genera** of common, vocal North American birds (`genus_list.json`), each
  with a representative common-name label and example species.
- **Quality A & B only** (`q:A`, `q:B`) — Xeno-canto's top two of five grades.
- **United States + Canada** (`cnt:"United States"`, `cnt:"Canada"`) to bias
  toward North American dialects. The API ANDs repeated tags, so each
  (country × quality) pair is queried separately and merged.
- **~100 recordings per genus**, **balanced across member species** (round-robin
  sampling) so one common/loud species can't dominate its genus class.

**Critical data-quality fix — genus prefix contamination.** Xeno-canto's `gen:`
search is a **prefix match**: `gen:Passer` also returns *Passerina*, *Passerella*,
*Passerculus*; `gen:Buteo` returns *Buteogallus*. Left unchecked this mislabels
whole classes (our *Passer* class was mostly buntings/sparrows and scored 12%).
Fix: every API result is filtered to `rec["gen"] == genus` exactly.

**Final dataset:** ~4,966 recordings, 50 genera, mean ~99 recordings/genus
(min ~94). Raw audio (~8.8 GB) and recordings are gitignored; metadata in
`recordings.csv`.

---

## 2. Cleaning & filtering (`bird_classifier.py preprocess`)

All audio resampled to **32 kHz mono**. Per-recording cleaning is deliberately
**light** — over-filtering removes real signal, and the model is trained to be
noise-robust via augmentation:

1. **DC-offset removal** (subtract mean).
2. **High-pass filter @ 150 Hz** (4th-order Butterworth) — removes wind/traffic
   handling rumble, the single biggest noise source. Cutoff kept low so
   low-pitched genera (dove coos ~400–600 Hz) survive.
3. **Peak normalization** to a consistent level.
4. *(optional)* **spectral-gating denoise** (`noisereduce`, `--denoise`) — off by
   default; it can smear transients and hurt accuracy.

### Window selection — band-limited signal detection

Recordings are sliced into **5 s windows with 50% overlap** (hop 2.5 s). Windows
are ranked **not by raw loudness** but by **bird-band signal strength**: a
band-pass to **1–11 kHz** (`SIGNAL_BAND`) measures energy where bird vocalizations
live, so wind/handling/rain (mostly low-freq or broadband) is rejected. The
**top 12 windows per recording** above the noise floor are kept (lower bound
1 kHz preserves dove/corvid/woodpecker harmonics). Selection uses the band-limited
signal, but the saved window is **full-band** so the spectrogram keeps all info.

### Spectrogram representation — PCEN

Each window → **PCEN** (per-channel energy normalization) mel-spectrogram:

| Param | Value |
|-------|-------|
| n_mels | 128 |
| n_fft | 1024 |
| hop_length | 512 |
| fmin / fmax | 150 Hz / 15 kHz |
| shape | **128 × 313** (float16) |

PCEN is chosen over log-mel-dB because it applies automatic gain control +
dynamic-range compression — more robust to the variable-gain, noisy field
recordings that dominate Xeno-canto. (Log-mel-dB available via `--repr logmel`.)

**Result:** ~31,586 spectrogram windows from ~4,965 recordings (~1.5 GB,
gitignored). Index with `recording_id` in `spectrograms.csv`.

---

## 3. Model architecture

- **Backbone:** EfficientNet-B0 (timm, ImageNet-pretrained), first conv adapted to
  **1-channel** spectrogram input (`in_chans=1`). Ensemble adds EfficientNet-B1.
- **Head:** standard timm classifier, 50-way output.
- Chosen for the accuracy/CPU-speed balance; the sweep confirmed the small B0 with
  strong regularization beats heavier B1/B2 at this data size.

---

## 4. Training (`bird_classifier.py train`)

**Leakage-free split.** TRAIN/VAL is split **at the recording level**
(`split_by_recording`, stratified by genus, 15% val) — windows from one recording
never straddle the split. We report **recording-level** accuracy (aggregate window
predictions per recording), the metric that matches deployment.

**Class balancing.** `WeightedRandomSampler` (inverse genus-window frequency), since
recording lengths — and thus windows/genus — vary.

**Normalization.** Global mean/std computed from a 3 k-window train sample and
baked into `class_mapping.json` so inference reproduces it exactly.

**Loss / optim.** CrossEntropy with **label smoothing 0.1**; **AdamW** + **cosine
annealing**; mixed-precision (fp16 autocast) on GPU.

### Data augmentation — intentionally light (~10% of samples)

Augmentation is applied to only **~10% of training samples** (`AUG_PROB = 0.10`)
and mixup to **~10% of batches** (`MIXUP_PROB = 0.10`); the rest stay clean. On
clean data this light touch outperformed heavy augmentation. The augmentation set
(`augment_spec`), all on the PCEN spectrogram:

- **Time-roll** (circular shift along time)
- **SpecAugment** — up to 2 frequency masks + 2 time masks
- **Gaussian noise** injection
- **Random gain**
- **Tempo** — time-axis stretch (≈0.85–1.15×) approximating speed change
- **Pitch** — small mel-axis shift
- **Background-sound mixing** — add a faint random other spectrogram (label
  unchanged) to simulate background birds/noise
- **mixup** — convex mix of sample pairs + labels (batch level)

---

## 5. Hyperparameter sweep (`bird_classifier.py sweep`)

Optimizes **recording-level top-1** (the deployed metric). All spectrograms are
cached in RAM (~3 GB) so each run avoids disk I/O. **8 configs** are screened at
25 epochs (B0/B1/B2 × learning-rate × weight-decay × mixup), then the winner is
retrained for 45 epochs. The sweep is **restartable** — each config checkpoints to
`sweep_results.csv` and is skipped on resume.

**Winner:** **EfficientNet-B0, lr 1e-3, weight-decay 0.05, mixup 0.2** — strong
regularization + high LR beat the larger backbones, a sign that at this data size
regularization matters more than capacity.

---

## 6. Ensemble (`bird_classifier.py ensemble`)

To push past the 80% top-1 target, the two best sweep configs are averaged:
**EfficientNet-B0 (strong-reg)** + **EfficientNet-B1 (lr 5e-4, no mixup)**.
Per-window softmax is averaged across both nets, then aggregated per recording.
Decorrelated backbones add ~1–2% over either alone. Both members are
ImageNet-pretrained and trained with the identical split/normalization.

---

## 7. Inference (`bird_classifier.py predict`, CPU)

```bash
python bird_classifier.py predict call.mp3
python bird_classifier.py predict call.mp3 --top-k 5
python bird_classifier.py predict call.mp3 --tta      # optional, ~2x cost
```

- Reproduces training preprocessing exactly (config read from
  `class_mapping.json`), slides windows over the whole recording, runs each model,
  and **aggregates per-window probabilities energy-weighted on the bird band** so
  quiet windows contribute little.
- **Non-overlapping windows by default** (lean CPU). `--tta` uses 50% overlap for a
  small accuracy bump at ~2× CPU cost.
- Auto-detects single model vs. ensemble from `class_mapping.json`.
- **Latency (CPU, warm):** ~0.2 s/recording single model, ~0.4 s ensemble — well
  within the 1–3 s budget. First call adds ~1–2 s of one-time numba warmup.

---

## 8. Results

Recording-level accuracy on the held-out (split **by recording**) validation set.
Two measurement bases are reported because they differ meaningfully:

- **Training-eval** — aggregate the top-12 energy-selected *precomputed* windows
  (the conservative metric used during the sweep).
- **Deployed** — the real `predict` path: window the **whole** recording and
  energy-weight **all** windows (745 val recordings). This is what users get.

| Model | basis | top-1 | top-3 |
|-------|-------|-------|-------|
| B0 — original (pre-fix, heavy aug) | training-eval | 0.751 | 0.861 |
| B0 — clean + filtered + light-aug + sweep winner | training-eval | 0.779 | 0.894 |
| B1 — sweep runner-up | training-eval | 0.781 | 0.874 |
| **Ensemble B0+B1** | training-eval | 0.795 | 0.903 |
| B0 — deployed | full path | 0.793 | 0.905 |
| **Ensemble B0+B1 — deployed (non-TTA, default)** | full path | **0.808** | **0.918** |
| Ensemble B0+B1 — deployed **+ TTA** | full path | 0.811 | 0.917 |

**The ensemble clears the 80% top-1 target at 80.8% (non-TTA) / 91.8% top-3** — so
the lean non-overlapping default ships; TTA adds only ~+0.3% here and is left off.

> Progression: contamination fix + band-limited window selection + light
> augmentation + sweep took B0 from 75.1% → 79.3% (deployed), and the B0+B1
> ensemble crossed 80% at **80.8%**.

---

## Files

| File | Purpose |
|------|---------|
| **`bird_classifier.py`** | the entire pipeline — `download`/`preprocess`/`train`/`sweep`/`ensemble`/`predict`/`tta-eval` |
| `genus_list.json` | the 50 genera (scientific + common labels + examples) |
| `requirements.txt` | Python deps |
| `eda.ipynb` | dataset EDA (plots embedded; data regenerable) |
| `model_b0.pth`, `model_b1.pth` | ensemble member weights (committed; 16 + 26 MB) |
| `label_vocabs.json`, `class_mapping.json` | labels + inference config/metadata (wired to the ensemble) |
| `recordings.csv` | dataset manifest (provenance of the recordings used) |

The raw `audio/` + `spectrograms/` training data was removed after training to keep
the folder lean (~62 MB); regenerate with `bird_classifier.py download` then
`preprocess`. Audio, spectrograms, weights, CSVs, logs, and the API key are
gitignored.
