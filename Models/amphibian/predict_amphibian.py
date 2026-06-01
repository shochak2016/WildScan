"""
predict_amphibian.py — Classify amphibian sounds (ONNX version, no TensorFlow needed).

Usage:
    python predict_amphibian.py recording.wav
    python predict_amphibian.py sounds/*.wav

Dependencies (all support modern Python):
    pip install onnxruntime librosa scikit-learn joblib numpy
"""

import os
import sys
import numpy as np
import librosa
import joblib
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

SAMPLE_RATE = 22050
DURATION = 10
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
MODEL_DIR = "amphibian_output"

FAMILY_EXAMPLES = {
    "Tree Frogs (Spring Peepers, Gray Treefrogs)": "Spring Peeper, Pacific Chorus Frog, Gray Treefrog, Cricket Frog",
    "True Frogs (Bullfrogs, Leopard Frogs)": "American Bullfrog, Green Frog, Wood Frog, Leopard Frog",
    "True Toads (American Toad, Cane Toad)": "American Toad, Fowler's Toad, Gulf Coast Toad",
    "Narrow-mouthed Frogs": "Eastern Narrow-mouthed Toad, Western Narrow-mouthed Toad",
    "Rain Frogs & Coquís": "Common Coqui, Rio Grande Chirping Frog, Greenhouse Frog",
}


def load_audio(filepath):
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE, duration=DURATION)
    target = SAMPLE_RATE * DURATION
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    else:
        y = y[:target]
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y, sr


def get_mel(filepath):
    y, sr = load_audio(filepath)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return mel_db


def get_audio_features(filepath):
    y, sr = load_audio(filepath)
    features = []

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))
    features.extend(np.max(mfcc, axis=1))
    features.extend(np.min(mfcc, axis=1))

    mfcc_delta = librosa.feature.delta(mfcc)
    features.extend(np.mean(mfcc_delta, axis=1))
    features.extend(np.std(mfcc_delta, axis=1))

    # chroma_stft segfaults under this numpy/BLAS build (hard crash); 24-dim
    # zero fallback (mean+std of 12 chroma bins) to preserve the feature shape.
    features.extend([0.0] * 24)

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features.extend(np.mean(contrast, axis=1))
    features.extend(np.std(contrast, axis=1))

    # NOTE: librosa.effects.harmonic / tonnetz segfaults under this numba build
    # (a hard crash, not a catchable exception), so we use the original code's
    # 12-zero fallback for the tonnetz block to keep the feature-vector shape.
    features.extend([0.0] * 12)

    for feat_fn in [
        librosa.feature.spectral_centroid,
        librosa.feature.spectral_bandwidth,
        librosa.feature.spectral_rolloff,
        librosa.feature.zero_crossing_rate,
        librosa.feature.rms,
    ]:
        if feat_fn == librosa.feature.zero_crossing_rate:
            arr = feat_fn(y)
        elif feat_fn == librosa.feature.rms:
            arr = feat_fn(y=y)
        else:
            arr = feat_fn(y=y, sr=sr)
        features.extend([np.mean(arr), np.std(arr), np.max(arr), np.min(arr)])

    S = np.abs(librosa.stft(y, n_fft=N_FFT))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    power = np.mean(S, axis=1)
    if power.sum() > 0:
        features.append(freqs[np.argmax(power)])
        features.append(np.sum(power * freqs) / np.sum(power))
        low = (freqs >= 500) & (freqs < 2000)
        mid = (freqs >= 2000) & (freqs < 5000)
        high = (freqs >= 5000) & (freqs < 10000)
        total = power.sum()
        features.append(power[low].sum() / total if total > 0 else 0)
        features.append(power[mid].sum() / total if total > 0 else 0)
        features.append(power[high].sum() / total if total > 0 else 0)
    else:
        features.extend([0.0] * 5)

    return np.array(features, dtype=np.float32)


def _fix_sklearn_compat(obj):
    """Fix sklearn version mismatches when loading old models on newer sklearn.
    Patches renamed attributes so models trained on 1.6.x work on 1.8.x."""
    import sklearn.impute._base as impute_base
    
    # Walk through pipelines and fix SimpleImputer objects
    if hasattr(obj, 'named_steps'):
        for step_name, step in obj.named_steps.items():
            _fix_sklearn_compat(step)
    if hasattr(obj, 'estimators_'):
        for est in obj.estimators_:
            if isinstance(est, tuple):
                _fix_sklearn_compat(est[1])
            else:
                _fix_sklearn_compat(est)
    
    # Fix _fill_dtype → _fit_dtype rename (sklearn 1.6 → 1.8)
    if hasattr(obj, '__class__') and obj.__class__.__name__ == 'SimpleImputer':
        if not hasattr(obj, '_fill_dtype') and hasattr(obj, '_fit_dtype'):
            obj._fill_dtype = obj._fit_dtype
    
    return obj


def load_models(model_dir):
    print("Loading models...")

    import onnxruntime as ort

    # Load ONNX embedding model (replaces TensorFlow)
    onnx_path = os.path.join(model_dir, "embedding_model.onnx")
    if not os.path.exists(onnx_path):
        print(f"ERROR: {onnx_path} not found!")
        print("Convert it on Colab first (see INSTRUCTIONS.md).")
        sys.exit(1)

    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    models = {
        "onnx_session": session,
        "onnx_input_name": input_name,
        "voting": _fix_sklearn_compat(joblib.load(os.path.join(model_dir, "voting_ensemble.joblib"))),
        "feat_pipe": _fix_sklearn_compat(joblib.load(os.path.join(model_dir, "audio_feat_preprocessor.joblib"))),
        "selector": joblib.load(os.path.join(model_dir, "feature_selector.joblib")),
        "labels": np.load(os.path.join(model_dir, "label_classes.npy"), allow_pickle=True),
    }

    print(f"Loaded! Classes: {list(models['labels'])}\n")
    return models


def predict(filepath, models):
    # 1. Mel spectrogram → ONNX embedding
    mel = get_mel(filepath)
    mel_input = mel[np.newaxis, ..., np.newaxis].astype(np.float32)

    embedding = models["onnx_session"].run(
        None, {models["onnx_input_name"]: mel_input}
    )[0]

    # 2. Handcrafted audio features
    audio_feats = get_audio_features(filepath).reshape(1, -1)
    audio_feats_scaled = models["feat_pipe"].transform(audio_feats)

    # 3. Combine
    combined = np.hstack([embedding, audio_feats_scaled])

    # Pad for metadata columns if needed
    expected = models["selector"].n_features_in_
    if combined.shape[1] < expected:
        padding = np.zeros((1, expected - combined.shape[1]))
        combined = np.hstack([combined, padding])

    # 4. Select features + predict
    selected = models["selector"].transform(combined)
    pred_idx = models["voting"].predict(selected)[0]

    # 5. Probabilities
    try:
        probs = models["voting"].predict_proba(selected)[0]
        confidence = probs[pred_idx]
        top3_idx = np.argsort(probs)[::-1][:3]
    except Exception:
        confidence = None
        probs = None
        top3_idx = [pred_idx]

    family = models["labels"][pred_idx]
    return family, confidence, top3_idx, probs


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_amphibian.py audio_file.wav")
        sys.exit(1)

    models = load_models(MODEL_DIR)

    for filepath in sys.argv[1:]:
        if not os.path.exists(filepath):
            print(f"  {filepath}: FILE NOT FOUND\n")
            continue

        try:
            family, confidence, top3_idx, probs = predict(filepath, models)
            examples = FAMILY_EXAMPLES.get(family, "")

            print(f"  {Path(filepath).name}")
            print(f"    Family     → {family}")
            if confidence:
                print(f"    Confidence → {confidence:.1%}")
            if examples:
                print(f"    Could be   → {examples}")
            if probs is not None and len(top3_idx) > 1:
                print(f"    Other possibilities:")
                for idx in top3_idx[1:]:
                    print(f"      {models['labels'][idx]} — {probs[idx]:.1%}")
            print()

        except Exception as e:
            print(f"  {filepath}: Error — {e}\n")


if __name__ == "__main__":
    main()
