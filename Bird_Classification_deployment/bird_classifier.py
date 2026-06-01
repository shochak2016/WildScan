#!/usr/bin/env python3
"""
WildScan Bird-Sound Classifier — entire pipeline in one file.

Classifies a bird recording into one of 50 North American bird GENERA. Trains on
GPU; the shipped model runs on CPU (~0.2-0.4 s/recording). See README.md for the
full writeup of dataset, cleaning, augmentation, sweep, and ensemble choices.

Subcommands
-----------
  download    Fetch Xeno-canto recordings for the genera in genus_list.json
  preprocess  Clean -> band-SNR window select -> PCEN spectrograms (.npy)
  train       Train a single model
  sweep       Restartable hyperparameter sweep + retrain winner
  ensemble    Build the B0+B1 ensemble and evaluate it
  predict     CPU inference on audio file(s)               <-- deployment entry
  tta-eval    Compare TTA vs non-TTA on the val split

Examples
--------
  export XC_API_KEY=...                       # or write Bird_..._deployment/.xc_api_key
  python bird_classifier.py download
  python bird_classifier.py preprocess
  python bird_classifier.py sweep
  python bird_classifier.py ensemble
  python bird_classifier.py predict call.mp3
  python bird_classifier.py predict call.mp3 --top-k 5 --tta

Requires: torch torchaudio torchvision timm librosa soundfile noisereduce scipy
          numpy pandas scikit-learn requests tqdm   (see requirements.txt)
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent

# ============================================================================
# Config — audio/spectrogram params are shared by preprocessing AND inference
# ============================================================================
SR = 32000                 # resample rate (Hz)
WINDOW_S = 5.0             # window length (s)
HOP_S = 2.5               # stride between training windows (50% overlap)
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128
FMIN = 150                # mel floor AND high-pass cutoff (kills rumble)
FMAX = 15000
TARGET_FRAMES = 1 + int(WINDOW_S * SR) // HOP_LENGTH          # 313
MAX_WINDOWS_PER_REC = 12
SIGNAL_BAND = (1000, 11000)  # bird-band used to rank windows by signal strength

AUG_PROB = 0.10            # fraction of TRAIN samples that get augmented
MIXUP_PROB = 0.10          # fraction of batches that get mixup

API_BASE = "https://xeno-canto.org/api/3/recordings"
USER_AGENT = "WildScan-bird-classifier/1.0 (research; contact founders@automorphic.ai)"
API_SLEEP_S, REQUEST_TIMEOUT = 1.0, 30

# Top-2 sweep configs (used by `ensemble`)
CFG_B0 = dict(model="efficientnet_b0", lr=1e-3, wd=0.05, mixup=0.2, specaug=True, bs=128)
CFG_B1 = dict(model="efficientnet_b1", lr=5e-4, wd=1e-4, mixup=0.0, specaug=True, bs=128)


# ============================================================================
# Xeno-canto download
# ============================================================================
def get_api_key():
    key = os.environ.get("XC_API_KEY", "").strip()
    if not key and (HERE / ".xc_api_key").exists():
        key = (HERE / ".xc_api_key").read_text().strip()
    if not key:
        sys.exit("ERROR: no Xeno-canto API key (set XC_API_KEY or .xc_api_key). "
                 "Get one free at https://xeno-canto.org/account")
    return key


def api_get(key, query, page=1):
    import requests
    r = requests.get(API_BASE, params={"query": query, "key": key, "page": page},
                     timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT})
    if r.status_code != 200:
        raise RuntimeError(f"API HTTP {r.status_code} for {query!r}: {r.text[:200]}")
    return r.json()


def fetch_genus(key, genus, qualities, countries, max_per_genus, max_pages=30):
    """Server-side filtered query per (country x quality), merged + deduped.
    gen: is a PREFIX match, so we keep only exact-genus rows (avoids e.g.
    gen:Passer pulling Passerina/Passerella, gen:Buteo pulling Buteogallus)."""
    collected = {}
    for country in countries:
        for q in qualities:
            query = f'gen:{genus} q:{q} cnt:"{country}"'
            try:
                first = api_get(key, query, 1)
            except RuntimeError as e:
                print(f"  [{genus}] {query!r} failed: {e}"); continue
            payloads = [first]
            for p in range(2, min(int(first.get("numPages", 1) or 1), max_pages) + 1):
                time.sleep(API_SLEEP_S)
                try:
                    payloads.append(api_get(key, query, p))
                except RuntimeError as e:
                    print(f"  [{genus}] {query!r} page {p} failed: {e}")
            for payload in payloads:
                for rec in payload.get("recordings", []):
                    if rec.get("gen") == genus:
                        collected[rec["id"]] = rec
            time.sleep(API_SLEEP_S)
    return balance_across_species(list(collected.values()), max_per_genus)


def balance_across_species(recs, cap):
    """Cap to `cap`, round-robin across species so one species can't dominate."""
    if len(recs) <= cap:
        return recs
    by_sp = {}
    for r in recs:
        by_sp.setdefault(r.get("sp", "?"), []).append(r)
    picked, pools, i = [], list(by_sp.values()), 0
    while len(picked) < cap and any(pools):
        pool = pools[i % len(pools)]
        if pool:
            picked.append(pool.pop())
        i += 1
        if i % len(by_sp) == 0:
            pools = [p for p in pools if p]
    return picked[:cap]


def download_one(rec, genus, audio_dir):
    import requests
    url = rec.get("file") or ""
    if url.startswith("//"):
        url = "https:" + url
    if not url:
        return None
    out = audio_dir / f"{genus}_{rec['id']}.mp3"
    if not out.exists():
        try:
            r = requests.get(url, timeout=REQUEST_TIMEOUT, stream=True,
                             headers={"User-Agent": USER_AGENT})
            if r.status_code != 200:
                return None
            with open(out, "wb") as f:
                for chunk in r.iter_content(65536):
                    f.write(chunk)
        except Exception:
            out.unlink(missing_ok=True)
            return None
    return {"id": rec["id"], "genus": genus, "species": rec.get("sp", ""),
            "english_name": rec.get("en", ""), "country": rec.get("cnt", ""),
            "quality": rec.get("q", ""), "type": rec.get("type", ""),
            "length": rec.get("length", ""), "license": rec.get("lic", ""),
            "file_url": url, "audio_path": str(out.relative_to(HERE))}


def cmd_download(args):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    key = get_api_key()
    if args.probe:
        payload = api_get(key, f"gen:{args.probe} q:A", 1)
        recs = payload.get("recordings", [])
        print("numRecordings:", payload.get("numRecordings"), "numPages:", payload.get("numPages"))
        if recs:
            print(json.dumps(recs[0], indent=2)[:1500])
        return
    data_dir = Path(args.data_dir)
    audio_dir = data_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    genera = [g["genus"] for g in json.load(open(HERE / "genus_list.json"))["genera"]]
    if args.limit_genera:
        genera = genera[:args.limit_genera]
    print(f"Genera: {len(genera)} | q={args.qualities} | {args.countries} | cap={args.max_per_genus}\n")
    rows = []
    for genus in genera:
        recs = fetch_genus(key, genus, args.qualities, args.countries, args.max_per_genus)
        got = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(download_one, r, genus, audio_dir) for r in recs]
            for f in tqdm(as_completed(futs), total=len(futs), desc=genus, leave=False):
                if f.result():
                    got.append(f.result())
        rows.extend(got)
        print(f"[{genus}] {len(got)} files")
    pd.DataFrame(rows).to_csv(data_dir / "recordings.csv", index=False)
    print(f"\nDone. {len(rows)} recordings -> recordings.csv")


# ============================================================================
# Cleaning + windowing + spectrogram (shared by preprocess AND predict)
# ============================================================================
def _butter(y, sr, kind, freqs, order=4):
    from scipy.signal import butter, sosfiltfilt
    sos = butter(order, freqs, btype=kind, fs=sr, output="sos")
    return sosfiltfilt(sos, y).astype(np.float32)


def highpass(y, sr, cutoff=FMIN):
    return _butter(y, sr, "highpass", cutoff)


def bandpass(y, sr, band=SIGNAL_BAND):
    low, high = band
    return _butter(y, sr, "bandpass", [low, min(high, sr / 2 - 100)])


def clean(y, sr, denoise=False):
    y = y - np.mean(y)
    y = highpass(y, sr)
    if denoise:
        import noisereduce as nr
        y = nr.reduce_noise(y=y, sr=sr, stationary=False, prop_decrease=0.6)
    peak = np.max(np.abs(y))
    return (y / peak).astype(np.float32) if peak > 1e-8 else y.astype(np.float32)


def select_windows(y, sr, max_windows=MAX_WINDOWS_PER_REC):
    """Rank windows by BIRD-BAND signal strength (rejects wind/handling); return
    the top full-band windows. Always returns >=1."""
    wlen, hop = int(WINDOW_S * sr), int(HOP_S * sr)
    if len(y) < wlen:
        y = np.pad(y, (0, wlen - len(y)))
    y_band = bandpass(y, sr)
    starts = list(range(0, max(1, len(y) - wlen + 1), hop))
    rms = np.array([np.sqrt(np.mean(y_band[s:s + wlen] ** 2)) for s in starts])
    floor = np.percentile(rms, 10)
    keep = [i for i in range(len(starts)) if rms[i] >= max(floor * 1.5, 1e-4)] or [int(np.argmax(rms))]
    keep = sorted(keep, key=lambda i: rms[i], reverse=True)[:max_windows]
    return [y[starts[i]:starts[i] + wlen] for i in sorted(keep)]


def spectrogram(y, sr, repr="pcen"):
    import librosa
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                          n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=1.0)
    if repr == "pcen":
        spec = librosa.pcen(mel * (2 ** 31), sr=sr, hop_length=HOP_LENGTH)
    else:
        spec = (librosa.power_to_db(mel ** 2, ref=np.max) + 80.0) / 80.0
    if spec.shape[1] < TARGET_FRAMES:
        spec = np.pad(spec, ((0, 0), (0, TARGET_FRAMES - spec.shape[1])))
    return spec[:, :TARGET_FRAMES].astype(np.float16)


def _process_recording(task):
    import librosa
    row, data_dir, spec_dir, repr, denoise = task
    path = Path(data_dir) / row["audio_path"]
    if not path.exists():
        return []
    try:
        y, _ = librosa.load(path, sr=SR, mono=True)
    except Exception:
        return []
    if y is None or len(y) < int(0.5 * SR):
        return []
    y = clean(y, SR, denoise)
    rows = []
    for wi, w in enumerate(select_windows(y, SR)):
        out = Path(spec_dir) / f"{row['genus']}_{row['id']}_{wi}.npy"
        np.save(out, spectrogram(w, SR, repr))
        rows.append({"spec_path": str(out.relative_to(HERE)), "genus": row["genus"],
                     "recording_id": row["id"], "species": row.get("species", "")})
    return rows


def cmd_preprocess(args):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm
    data_dir = Path(args.data_dir)
    spec_dir = data_dir / "spectrograms"
    spec_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_dir / "recordings.csv")
    if args.limit:
        df = df.head(args.limit)
    print(f"Recordings: {len(df)} | repr={args.repr} | denoise={args.denoise} | workers={args.workers}")
    tasks = [(r, str(data_dir), str(spec_dir), args.repr, args.denoise) for _, r in df.iterrows()]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for fut in tqdm([ex.submit(_process_recording, t) for t in tasks], desc="preprocess"):
            rows.extend(fut.result())
    out = pd.DataFrame(rows)
    out.to_csv(data_dir / "spectrograms.csv", index=False)
    print(f"\nDone. {len(out)} windows from {out['recording_id'].nunique()} recordings, "
          f"{out['genus'].nunique()} genera -> spectrograms.csv")


# ============================================================================
# Augmentation + dataset
# ============================================================================
def augment_spec(spec, specaug=True, tempo=True, pitch=True, gain=True, noise_std=0.08):
    """Mild PCEN-spectrogram augmentations: tempo (time-stretch), pitch (mel-shift),
    time-roll, gain, Gaussian noise, SpecAugment masks."""
    T0 = spec.shape[1]
    if tempo and np.random.rand() < 0.5:
        from scipy.ndimage import zoom
        spec = zoom(spec, (1, np.random.uniform(0.85, 1.15)), order=1)
        spec = spec[:, :T0] if spec.shape[1] >= T0 else np.pad(spec, ((0, 0), (0, T0 - spec.shape[1])))
    if pitch and np.random.rand() < 0.5:
        sh = np.random.randint(-8, 9)
        spec = np.roll(spec, sh, axis=0)
        if sh > 0: spec[:sh, :] = 0.0
        elif sh < 0: spec[sh:, :] = 0.0
    spec = np.roll(spec, np.random.randint(spec.shape[1]), axis=1)
    if gain:
        spec = spec * np.random.uniform(0.8, 1.2)
    if noise_std > 0:
        spec = spec + np.random.randn(*spec.shape).astype(np.float32) * noise_std
    if specaug:
        for _ in range(np.random.randint(0, 3)):
            f = np.random.randint(0, 20); f0 = np.random.randint(0, max(1, spec.shape[0] - f))
            spec[f0:f0 + f, :] = 0.0
        for _ in range(np.random.randint(0, 3)):
            t = np.random.randint(0, 40); t0 = np.random.randint(0, max(1, spec.shape[1] - t))
            spec[:, t0:t0 + t] = 0.0
    return spec.astype(np.float32)


class SpecDataset:
    """Spectrogram dataset (torch.utils.data.Dataset). Reads from a RAM `cache`
    dict if given, else from disk. Augments only ~AUG_PROB of TRAIN samples."""
    def __init__(self, df, l2i, root, mean, std, train=False, specaug=True, cache=None):
        import torch  # noqa
        self.df = df.reset_index(drop=True)
        self.l2i, self.root, self.mean, self.std = l2i, Path(root), mean, std
        self.train, self.specaug, self.cache = train, specaug, cache

    def __len__(self):
        return len(self.df)

    def _load(self, i):
        p = self.df.iloc[i]["spec_path"]
        return (self.cache[p] if self.cache is not None
                else np.load(self.root / p)).astype(np.float32)

    def __getitem__(self, i):
        import torch
        spec = self._load(i)
        if self.train and np.random.rand() < AUG_PROB:
            if np.random.rand() < 0.5:                       # background-sound mixing
                spec = spec + np.random.uniform(0.1, 0.4) * self._load(np.random.randint(len(self.df)))
            spec = augment_spec(spec, specaug=self.specaug)
        spec = (spec - self.mean) / (self.std + 1e-6)
        return torch.from_numpy(spec).unsqueeze(0), self.l2i[self.df.iloc[i]["genus"]]


# ============================================================================
# Training utilities
# ============================================================================
def split_by_recording(df, val_frac=0.15, seed=42):
    from sklearn.model_selection import train_test_split
    recs = df[["recording_id", "genus"]].drop_duplicates("recording_id")
    tr, va = train_test_split(recs, test_size=val_frac, stratify=recs["genus"], random_state=seed)
    return (df[df["recording_id"].isin(set(tr["recording_id"]))].copy(),
            df[df["recording_id"].isin(set(va["recording_id"]))].copy())


def compute_norm_stats(df, root, cache=None, n=3000, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    load = (lambda p: cache[p]) if cache is not None else (lambda p: np.load(Path(root) / p))
    arr = np.stack([load(df.iloc[i]["spec_path"]).astype(np.float32) for i in idx])
    return float(arr.mean()), float(arr.std())


def balanced_sampler(df, l2i):
    from torch.utils.data import WeightedRandomSampler
    counts = df["genus"].value_counts().to_dict()
    w = df["genus"].map(lambda g: 1.0 / counts[g]).values
    return WeightedRandomSampler(w, num_samples=len(df), replacement=True)


def build_model(arch, num_classes, device, pretrained=True):
    import timm
    return timm.create_model(arch, pretrained=pretrained, num_classes=num_classes, in_chans=1).to(device)


def _autocast(device):
    import torch
    return torch.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda"))


def member_probs_for(model, df, root, mean, std, device, cache=None):
    """Per-window softmax for every row of df (in order)."""
    import torch
    load = (lambda p: cache[p]) if cache is not None else (lambda p: np.load(Path(root) / p))
    specs = (np.stack([load(p) for p in df["spec_path"]]).astype(np.float32) - mean) / (std + 1e-6)
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(specs), 1024):
            x = torch.from_numpy(specs[i:i + 1024][:, None]).float().to(device)
            with _autocast(device):
                out.append(torch.softmax(model(x), 1).float().cpu().numpy())
    return np.concatenate(out)


def recording_metrics(probs, df, l2i):
    """Aggregate window probs per recording -> (top1, top3)."""
    df = df.reset_index(drop=True)
    t1 = t3 = n = 0
    for _, grp in df.groupby("recording_id"):
        agg = probs[grp.index.values].mean(0)
        true = l2i[grp["genus"].iloc[0]]
        t1 += int(agg.argmax() == true)
        t3 += int(true in np.argsort(agg)[-3:])
        n += 1
    return t1 / n, t3 / n


def train_one(cfg, epochs, data, device, save_path=None, verbose=False):
    """Train one model; track best recording-level top1; optionally save best."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    train_df, val_df, cache, l2i, root, mean, std = data
    model = build_model(cfg["model"], len(l2i), device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    ds = SpecDataset(train_df, l2i, root, mean, std, train=True, specaug=cfg["specaug"], cache=cache)
    loader = DataLoader(ds, batch_size=cfg["bs"], sampler=balanced_sampler(train_df, l2i),
                        num_workers=6, pin_memory=True, drop_last=True, persistent_workers=True)
    best = (0.0, 0.0)
    for ep in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            with _autocast(device):
                if cfg["mixup"] > 0 and np.random.rand() < MIXUP_PROB:
                    lam = np.random.beta(cfg["mixup"], cfg["mixup"])
                    perm = torch.randperm(x.size(0), device=device)
                    x = lam * x + (1 - lam) * x[perm]
                    out = model(x)
                    loss = lam * crit(out, y) + (1 - lam) * crit(out, y[perm])
                else:
                    loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
        t1, t3 = recording_metrics(member_probs_for(model, val_df, root, mean, std, device, cache), val_df, l2i)
        if verbose:
            print(f"  epoch {ep+1}/{epochs} | rec top1 {t1:.3f} top3 {t3:.3f}"
                  + ("  *best" if t1 > best[0] else ""))
        if t1 > best[0]:
            best = (t1, t3)
            if save_path:
                torch.save(model.state_dict(), save_path)
    return best


def prepare_data(data_dir, ram_cache=True, seed=42):
    df = pd.read_csv(Path(data_dir) / "spectrograms.csv")
    genera = sorted(df["genus"].unique())
    l2i = {g: i for i, g in enumerate(genera)}
    cache = None
    if ram_cache:
        print("Caching spectrograms in RAM ...")
        cache = {p: np.load(Path(data_dir) / p) for p in df["spec_path"]}
    train_df, val_df = split_by_recording(df, seed=seed)
    mean, std = compute_norm_stats(train_df, data_dir, cache)
    print(f"Train {len(train_df)} / Val {len(val_df)} | {len(genera)} genera | norm {mean:.3f}/{std:.3f}")
    return df, genera, l2i, train_df, val_df, cache, mean, std


def save_metadata(data_dir, genera, model_arch, mean, std, top1, models=None, extra=None):
    data_dir = Path(data_dir)
    meta = {g["genus"]: g for g in json.load(open(HERE / "genus_list.json"))["genera"]}
    json.dump({
        "genera": genera,
        "genera_common": {g: meta.get(g, {}).get("common_label", g) for g in genera},
        "genera_examples": {g: meta.get(g, {}).get("examples", "") for g in genera},
    }, open(data_dir / "label_vocabs.json", "w"), indent=2)
    cm = {
        "model_architecture": model_arch, "num_classes": len(genera),
        "best_val_top1": top1, "norm_mean": mean, "norm_std": std,
        "spectrogram": {"sr": SR, "n_fft": N_FFT, "hop_length": HOP_LENGTH, "n_mels": N_MELS,
                        "fmin": FMIN, "fmax": FMAX, "window_s": WINDOW_S,
                        "target_frames": TARGET_FRAMES, "repr": "pcen"},
    }
    if models:
        cm["models"] = models
    if extra:
        cm.update(extra)
    json.dump(cm, open(data_dir / "class_mapping.json", "w"), indent=2)


# ============================================================================
# Commands: train / sweep / ensemble
# ============================================================================
def cmd_train(args):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df, genera, l2i, tr, va, cache, mean, std = prepare_data(args.data_dir, ram_cache=not args.no_cache)
    data = (tr, va, cache, l2i, args.data_dir, mean, std)
    cfg = dict(model=args.model, lr=args.lr, wd=args.weight_decay, mixup=0.0 if args.no_mixup else args.mixup_alpha,
               specaug=not args.no_specaug, bs=args.batch_size)
    t1, t3 = train_one(cfg, args.epochs, data, device, save_path=Path(args.data_dir) / "best_model.pth", verbose=True)
    print(f"\nBest recording-level: top1 {t1:.3f} top3 {t3:.3f}")
    save_metadata(args.data_dir, genera, args.model, mean, std, t1)
    print("Saved best_model.pth + metadata")


def cmd_sweep(args):
    import torch
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df, genera, l2i, tr, va, cache, mean, std = prepare_data(args.data_dir, ram_cache=True)
    data = (tr, va, cache, l2i, args.data_dir, mean, std)
    base = dict(model="efficientnet_b0", lr=3e-4, wd=1e-4, mixup=0.2, specaug=True, bs=128)
    C = lambda **kw: {**base, **kw}
    configs = [C(), C(mixup=0.0), C(mixup=0.0, lr=5e-4), C(model="efficientnet_b2"),
               C(model="efficientnet_b2", mixup=0.0, lr=5e-4),
               C(model="efficientnet_b2", mixup=0.1, lr=5e-4, wd=1e-2),
               C(model="efficientnet_b1", mixup=0.0, lr=5e-4), C(lr=1e-3, wd=5e-2)]
    res_path = Path(args.data_dir) / "sweep_results.csv"
    key = lambda c: f"{c['model']}|{c['lr']}|{c['wd']}|{c['mixup']}|{int(bool(c['specaug']))}|{c['bs']}"
    results, done = [], set()
    if res_path.exists():
        for _, r in pd.read_csv(res_path).iterrows():
            results.append(r.to_dict()); done.add(key(r))
        print(f"Resuming: {len(done)} config(s) cached")
    for i, cfg in enumerate(configs):
        if key(cfg) in done:
            print(f"[{i+1}/{len(configs)}] cached, skip"); continue
        t = time.time()
        t1, t3 = train_one(cfg, args.screen_epochs, data, device)
        print(f"[{i+1}/{len(configs)}] {cfg['model']} lr{cfg['lr']} wd{cfg['wd']} mix{cfg['mixup']} "
              f"-> top1 {t1:.3f} top3 {t3:.3f} ({time.time()-t:.0f}s)")
        results.append({**cfg, "rec_top1": t1, "rec_top3": t3})
        pd.DataFrame(results).to_csv(res_path, index=False)
    res_df = pd.DataFrame(results).sort_values("rec_top1", ascending=False)
    res_df.to_csv(res_path, index=False)
    print("\n", res_df[["model", "lr", "wd", "mixup", "rec_top1", "rec_top3"]].to_string(index=False))
    b = res_df.iloc[0].to_dict()
    win = dict(model=b["model"], lr=float(b["lr"]), wd=float(b["wd"]),
               mixup=float(b["mixup"]), specaug=bool(b["specaug"]), bs=int(b["bs"]))
    print(f"\nRetraining winner {win['model']} for {args.final_epochs} epochs ...")
    t1, t3 = train_one(win, args.final_epochs, data, device, save_path=Path(args.data_dir) / "best_model.pth")
    print(f"FINAL recording-level: top1 {t1:.3f} top3 {t3:.3f}")
    save_metadata(args.data_dir, genera, win["model"], mean, std, t1)


def cmd_ensemble(args):
    import shutil
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dd = Path(args.data_dir)
    df, genera, l2i, tr, va, cache, mean, std = prepare_data(args.data_dir, ram_cache=True)
    va = va.reset_index(drop=True)
    data = (tr, va, cache, l2i, args.data_dir, mean, std)
    if args.retrain_b0 or not (dd / "best_model.pth").exists():
        print("Training B0 ..."); train_one(CFG_B0, args.epochs, data, device, save_path=dd / "model_b0.pth")
    else:
        print("Reusing best_model.pth as model_b0.pth"); shutil.copy(dd / "best_model.pth", dd / "model_b0.pth")
    print("Training B1 ..."); train_one(CFG_B1, args.epochs, data, device, save_path=dd / "model_b1.pth")
    m0 = build_model("efficientnet_b0", 50, device, pretrained=False)
    m0.load_state_dict(torch.load(dd / "model_b0.pth", map_location=device))
    m1 = build_model("efficientnet_b1", 50, device, pretrained=False)
    m1.load_state_dict(torch.load(dd / "model_b1.pth", map_location=device))
    p0 = member_probs_for(m0, va, args.data_dir, mean, std, device, cache)
    p1 = member_probs_for(m1, va, args.data_dir, mean, std, device, cache)
    e0, e1, ens = (recording_metrics(p, va, l2i) for p in (p0, p1, (p0 + p1) / 2))
    print(f"\n  B0       top1 {e0[0]:.3f} top3 {e0[1]:.3f}")
    print(f"  B1       top1 {e1[0]:.3f} top3 {e1[1]:.3f}")
    print(f"  ENSEMBLE top1 {ens[0]:.3f} top3 {ens[1]:.3f}")
    save_metadata(args.data_dir, genera, "ensemble:b0+b1", mean, std, ens[0],
                  models=[{"file": "model_b0.pth", "arch": "efficientnet_b0"},
                          {"file": "model_b1.pth", "arch": "efficientnet_b1"}],
                  extra={"ensemble_member_top1": {"b0": e0[0], "b1": e1[0]}})
    print("Saved ensemble models + metadata")


# ============================================================================
# Inference (CPU)
# ============================================================================
def load_models(data_dir, device):
    cfg = json.load(open(Path(data_dir) / "class_mapping.json"))
    vocabs = json.load(open(Path(data_dir) / "label_vocabs.json"))
    specs = cfg.get("models") or [{"file": "best_model.pth", "arch": cfg["model_architecture"]}]
    models = []
    import torch
    for m in specs:
        net = build_model(m["arch"], cfg["num_classes"], device, pretrained=False)
        net.load_state_dict(torch.load(Path(data_dir) / m["file"], map_location=device))
        net.eval()
        models.append(net)
    return models, cfg, vocabs


def file_to_windows(audio_path, cfg, tta=False):
    import librosa
    sr = cfg["spectrogram"]["sr"]
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    y = clean(y, sr)
    wlen = int(cfg["spectrogram"]["window_s"] * sr)
    hop = wlen // 2 if tta else wlen
    if len(y) < wlen:
        y = np.pad(y, (0, wlen - len(y)))
    y_band = bandpass(y, sr)
    specs, energies = [], []
    for s in range(0, len(y) - wlen + 1, hop):
        specs.append(spectrogram(y[s:s + wlen], sr, cfg["spectrogram"]["repr"]).astype(np.float32))
        energies.append(float(np.sqrt(np.mean(y_band[s:s + wlen] ** 2))))
    return specs, np.array(energies)


def predict(audio_path, models, cfg, vocabs, device, top_k=3, tta=False):
    import torch
    import torch.nn.functional as F
    specs, energies = file_to_windows(audio_path, cfg, tta)
    mean, std = cfg["norm_mean"], cfg["norm_std"]
    x = torch.from_numpy(np.stack([(s - mean) / (std + 1e-6) for s in specs])[:, None]).float().to(device)
    with torch.no_grad():
        probs = np.mean([F.softmax(net(x), 1).cpu().numpy() for net in models], axis=0)
    agg = (probs * (energies / (energies.sum() + 1e-8))[:, None]).sum(0)
    genera = vocabs["genera"]
    order = np.argsort(agg)[::-1][:top_k]
    return [{"genus": genera[i], "common_label": vocabs["genera_common"].get(genera[i], genera[i]),
             "examples": vocabs["genera_examples"].get(genera[i], ""),
             "confidence": float(agg[i])} for i in order], len(specs)


def cmd_predict(args):
    import torch
    device = torch.device("cpu")
    models, cfg, vocabs = load_models(args.data_dir, device)
    print(f"Model: {cfg['model_architecture']} ({len(models)} net) | {cfg['num_classes']} genera "
          f"| val top1 {cfg.get('best_val_top1', 0):.1%}\n")
    for path in args.audio:
        if not Path(path).exists():
            print(f"  {path}: NOT FOUND\n"); continue
        preds, n = predict(path, models, cfg, vocabs, device, args.top_k, args.tta)
        best = preds[0]
        tag = "high" if best["confidence"] >= 0.70 else "medium" if best["confidence"] >= 0.45 else "low"
        print(f"  {Path(path).name}  ({n} windows)")
        print(f"    -> {best['common_label']} ({best['genus']}) — {best['confidence']:.1%} [{tag}]")
        if best["examples"]:
            print(f"       e.g. {best['examples']}")
        for p in preds[1:]:
            print(f"      {p['common_label']} ({p['genus']}) — {p['confidence']:.1%}")
        print()


# ============================================================================
# TTA evaluation
# ============================================================================
_TTA = {}


def _tta_init(data_dir):
    import torch
    torch.set_num_threads(1)
    _TTA["m"], _TTA["cfg"], _TTA["v"] = load_models(data_dir, torch.device("cpu"))


def _tta_work(task):
    import torch
    rid, true, path = task
    res = {}
    for tta in (False, True):
        preds, _ = predict(path, _TTA["m"], _TTA["cfg"], _TTA["v"], torch.device("cpu"), 3, tta)
        top3 = [p["genus"] for p in preds]
        res[tta] = (top3[0] == true, true in top3)
    return res


def cmd_tta_eval(args):
    from concurrent.futures import ProcessPoolExecutor
    from tqdm import tqdm
    dd = Path(args.data_dir)
    df = pd.read_csv(dd / "spectrograms.csv")
    _, val = split_by_recording(df, seed=42)
    recmap = {r["id"]: r["audio_path"] for _, r in pd.read_csv(dd / "recordings.csv").iterrows()}
    vr = val[["recording_id", "genus"]].drop_duplicates("recording_id")
    tasks = [(r["recording_id"], r["genus"], str(dd / recmap[r["recording_id"]]))
             for _, r in vr.iterrows() if r["recording_id"] in recmap]
    agg, n = {False: [0, 0], True: [0, 0]}, 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_tta_init, initargs=(args.data_dir,)) as ex:
        for res in tqdm(ex.map(_tta_work, tasks), total=len(tasks), desc="tta-eval"):
            n += 1
            for tta in (False, True):
                agg[tta][0] += res[tta][0]; agg[tta][1] += res[tta][1]
    print(f"\nval recordings: {n}")
    print(f"NON-TTA  top1 {agg[False][0]/n:.4f}  top3 {agg[False][1]/n:.4f}")
    print(f"TTA      top1 {agg[True][0]/n:.4f}  top3 {agg[True][1]/n:.4f}")
    print(f"DELTA    top1 {(agg[True][0]-agg[False][0])/n:+.4f}  top3 {(agg[True][1]-agg[False][1])/n:+.4f}")


# ============================================================================
# CLI
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="WildScan bird-sound classifier (one-file pipeline)")
    ap.add_argument("--data-dir", default=os.environ.get("XC_DATA_DIR", str(HERE)))
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download"); d.set_defaults(func=cmd_download)
    d.add_argument("--probe", metavar="GENUS"); d.add_argument("--max-per-genus", type=int, default=100)
    d.add_argument("--qualities", nargs="+", default=["A", "B"])
    d.add_argument("--countries", nargs="+", default=["United States", "Canada"])
    d.add_argument("--limit-genera", type=int); d.add_argument("--workers", type=int, default=8)

    p = sub.add_parser("preprocess"); p.set_defaults(func=cmd_preprocess)
    p.add_argument("--repr", choices=["pcen", "logmel"], default="pcen")
    p.add_argument("--denoise", action="store_true"); p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))

    t = sub.add_parser("train"); t.set_defaults(func=cmd_train)
    t.add_argument("--epochs", type=int, default=45); t.add_argument("--batch-size", type=int, default=128)
    t.add_argument("--lr", type=float, default=1e-3); t.add_argument("--weight-decay", type=float, default=0.05)
    t.add_argument("--model", default="efficientnet_b0"); t.add_argument("--mixup-alpha", type=float, default=0.2)
    t.add_argument("--no-mixup", action="store_true"); t.add_argument("--no-specaug", action="store_true")
    t.add_argument("--no-cache", action="store_true")

    s = sub.add_parser("sweep"); s.set_defaults(func=cmd_sweep)
    s.add_argument("--screen-epochs", type=int, default=25); s.add_argument("--final-epochs", type=int, default=45)

    e = sub.add_parser("ensemble"); e.set_defaults(func=cmd_ensemble)
    e.add_argument("--epochs", type=int, default=45); e.add_argument("--retrain-b0", action="store_true")

    pr = sub.add_parser("predict"); pr.set_defaults(func=cmd_predict)
    pr.add_argument("audio", nargs="+"); pr.add_argument("--top-k", type=int, default=3)
    pr.add_argument("--tta", action="store_true")

    te = sub.add_parser("tta-eval"); te.set_defaults(func=cmd_tta_eval)
    te.add_argument("--workers", type=int, default=18)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
