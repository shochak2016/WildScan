import os
import hashlib
import random
import warnings
import requests
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import librosa
import soundfile as sf

import tensorflow as tf
import tensorflow_hub as hub

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


# -------------------------
# CONFIG
# -------------------------

CSV_PATH = "birds.csv"

RAW_AUDIO_DIR = "audio_files"
WAV_AUDIO_DIR = "yamnet_wav_files"

EMBEDDING_CACHE = "yamnet_embeddings_cache.npz"

YAMNET_URL = "https://tfhub.dev/google/yamnet/1"

# YAMNet expects 16 kHz mono audio
YAMNET_SAMPLE_RATE = 16000

# Use the loudest 5 seconds so we are more likely to capture the bird call
CLIP_SECONDS = 5
CLIP_SAMPLES = YAMNET_SAMPLE_RATE * CLIP_SECONDS

SMALL_TEST = False

if SMALL_TEST:
    NUM_GENERA = 3
    MAX_PER_CLASS = 80
    MIN_PER_CLASS = 20
else:
    NUM_GENERA = 8
    MAX_PER_CLASS = 250
    MIN_PER_CLASS = 40

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
warnings.filterwarnings("ignore")


# -------------------------
# FILE HELPERS
# -------------------------

def url_to_filename(url):
    hashed = hashlib.md5(str(url).encode()).hexdigest()
    return f"{hashed}.mp3"


def wav_filename_from_url(url):
    hashed = hashlib.md5(str(url).encode()).hexdigest()
    return f"{hashed}.wav"


def download_audio(url, save_path):
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
        return True

    try:
        response = requests.get(url, timeout=20, allow_redirects=True)

        if response.status_code != 200:
            return False

        content_type = response.headers.get("Content-Type", "").lower()

        if "text/html" in content_type:
            return False

        with open(save_path, "wb") as f:
            f.write(response.content)

        if os.path.getsize(save_path) < 1000:
            os.remove(save_path)
            return False

        return True

    except Exception:
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
            except Exception:
                pass
        return False


def crop_loudest_window(y):
    if len(y) <= CLIP_SAMPLES:
        pad_width = CLIP_SAMPLES - len(y)
        return np.pad(y, (0, pad_width))

    hop = YAMNET_SAMPLE_RATE // 2
    best_start = 0
    best_energy = -1

    for start in range(0, len(y) - CLIP_SAMPLES, hop):
        window = y[start:start + CLIP_SAMPLES]
        energy = np.mean(window ** 2)

        if energy > best_energy:
            best_energy = energy
            best_start = start

    return y[best_start:best_start + CLIP_SAMPLES]


def convert_to_yamnet_wav(input_path, output_path):
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        return True

    try:
        y, sr = librosa.load(input_path, sr=YAMNET_SAMPLE_RATE, mono=True)

        if y is None or len(y) == 0:
            return False

        # Trim long silence
        try:
            y_trimmed, _ = librosa.effects.trim(y, top_db=35)
            if len(y_trimmed) > YAMNET_SAMPLE_RATE:
                y = y_trimmed
        except Exception:
            pass

        # Loudest-window crop
        y = crop_loudest_window(y)

        # Normalize
        max_abs = np.max(np.abs(y))
        if max_abs > 0:
            y = 0.9 * y / max_abs

        sf.write(output_path, y, YAMNET_SAMPLE_RATE)
        return True

    except Exception:
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except Exception:
                pass
        return False


# -------------------------
# YAMNET EMBEDDINGS
# -------------------------

def load_yamnet():
    print("\nLoading YAMNet from TensorFlow Hub...")
    model = hub.load(YAMNET_URL)
    print("YAMNet loaded.")
    return model


def extract_yamnet_embedding(yamnet_model, wav_path):
    y, sr = librosa.load(wav_path, sr=YAMNET_SAMPLE_RATE, mono=True)

    if y is None or len(y) == 0:
        return None

    y = y.astype(np.float32)

    # YAMNet returns scores, embeddings, spectrogram
    scores, embeddings, spectrogram = yamnet_model(y)

    embeddings = embeddings.numpy()

    # Average across time frames
    embedding_mean = embeddings.mean(axis=0)

    return embedding_mean


# -------------------------
# DATA PREP
# -------------------------

def prepare_dataframe():
    df = pd.read_csv(CSV_PATH)

    required_cols = ["sound_url", "latitude", "longitude", "scientific_name"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV must contain column: {col}")

    df = df.dropna(subset=["sound_url", "latitude", "longitude", "scientific_name"]).copy()

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    df = df.dropna(subset=["latitude", "longitude"])
    df = df[(df["latitude"].between(-90, 90)) & (df["longitude"].between(-180, 180))]

    df = df[df["sound_url"].astype(str).str.startswith("http")].copy()

    # Genus = first word of scientific name
    df["genus"] = df["scientific_name"].astype(str).str.split().str[0]
    df["label"] = df["genus"]

    top_labels = df["label"].value_counts().head(NUM_GENERA).index
    df = df[df["label"].isin(top_labels)].copy()

    df = (
        df.groupby("label", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), MAX_PER_CLASS), random_state=RANDOM_SEED))
        .reset_index(drop=True)
    )

    print(f"\nSMALL_TEST = {SMALL_TEST}")
    print(f"Using top {NUM_GENERA} genera capped at {MAX_PER_CLASS} each:")
    print(df["label"].value_counts())

    return df


def build_audio_files(df):
    os.makedirs(RAW_AUDIO_DIR, exist_ok=True)
    os.makedirs(WAV_AUDIO_DIR, exist_ok=True)

    keep_rows = []
    wav_paths = []

    print("\nDownloading/converting audio files for YAMNet...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        url = str(row["sound_url"])

        raw_path = os.path.join(RAW_AUDIO_DIR, url_to_filename(url))
        wav_path = os.path.join(WAV_AUDIO_DIR, wav_filename_from_url(url))

        downloaded = download_audio(url, raw_path)
        if not downloaded:
            continue

        converted = convert_to_yamnet_wav(raw_path, wav_path)
        if not converted:
            continue

        keep_rows.append(idx)
        wav_paths.append(wav_path)

    df = df.loc[keep_rows].copy()
    df["wav_path"] = wav_paths

    print(f"\nUsable WAV files: {len(df)}")
    print("\nLabel counts after conversion:")
    print(df["label"].value_counts())

    valid_labels = df["label"].value_counts()
    valid_labels = valid_labels[valid_labels >= MIN_PER_CLASS].index
    df = df[df["label"].isin(valid_labels)].copy()

    print(f"\nClasses kept after MIN_PER_CLASS={MIN_PER_CLASS}: {df['label'].nunique()}")
    print(df["label"].value_counts())

    if df["label"].nunique() < 2:
        raise ValueError("Need at least 2 classes after filtering.")

    return df


def build_embeddings(df):
    if os.path.exists(EMBEDDING_CACHE):
        print(f"\nLoading cached embeddings from {EMBEDDING_CACHE}...")
        data = np.load(EMBEDDING_CACHE, allow_pickle=True)

        X_audio = data["X_audio"]
        labels = data["labels"]
        latitudes = data["latitudes"]
        longitudes = data["longitudes"]

        print("Loaded cached embeddings:", X_audio.shape)

        return X_audio, labels, latitudes, longitudes

    yamnet_model = load_yamnet()

    embeddings = []
    labels = []
    latitudes = []
    longitudes = []

    print("\nExtracting YAMNet embeddings...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        emb = extract_yamnet_embedding(yamnet_model, row["wav_path"])

        if emb is None:
            continue

        embeddings.append(emb)
        labels.append(row["label"])
        latitudes.append(row["latitude"])
        longitudes.append(row["longitude"])

    X_audio = np.vstack(embeddings)
    labels = np.array(labels)
    latitudes = np.array(latitudes, dtype=np.float32)
    longitudes = np.array(longitudes, dtype=np.float32)

    np.savez(
        EMBEDDING_CACHE,
        X_audio=X_audio,
        labels=labels,
        latitudes=latitudes,
        longitudes=longitudes
    )

    print(f"\nSaved embedding cache to {EMBEDDING_CACHE}")
    print("Embedding shape:", X_audio.shape)

    return X_audio, labels, latitudes, longitudes


# -------------------------
# TRAIN CLASSIFIERS
# -------------------------

def train_and_evaluate(X_audio, labels, latitudes, longitudes):
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    print("\nClass mapping:")
    for i, label in enumerate(label_encoder.classes_):
        print(i, "=", label)

    # Normalize coordinates
    lat_mean = latitudes.mean()
    lat_std = latitudes.std() + 1e-6
    lon_mean = longitudes.mean()
    lon_std = longitudes.std() + 1e-6

    lat_norm = ((latitudes - lat_mean) / lat_std).reshape(-1, 1)
    lon_norm = ((longitudes - lon_mean) / lon_std).reshape(-1, 1)

    X_geo = np.hstack([lat_norm, lon_norm])

    # Audio-only features
    X_audio_only = X_audio

    # Audio + coordinates
    X_audio_geo = np.hstack([X_audio, X_geo])

    for name, X in [
        ("YAMNet audio only", X_audio_only),
        ("YAMNet audio + coordinates", X_audio_geo)
    ]:
        print("\n" + "=" * 60)
        print(name)
        print("=" * 60)

        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=RANDOM_SEED,
            stratify=y
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            random_state=RANDOM_SEED,
            stratify=y_temp
        )

        print("Train:", len(X_train))
        print("Val:", len(X_val))
        print("Test:", len(X_test))

        # Logistic Regression
        logreg = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                C=1.0,
                solver="lbfgs"
            )
        )

        logreg.fit(X_train, y_train)

        val_pred = logreg.predict(X_val)
        test_pred = logreg.predict(X_test)

        print("\nLogistic Regression")
        print("Val Accuracy:", accuracy_score(y_val, val_pred))
        print("Test Accuracy:", accuracy_score(y_test, test_pred))
        print(
            classification_report(
                y_test,
                test_pred,
                target_names=label_encoder.classes_,
                zero_division=0
            )
        )

        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_SEED
        )

        rf.fit(X_train, y_train)

        val_pred = rf.predict(X_val)
        test_pred = rf.predict(X_test)

        print("\nRandom Forest")
        print("Val Accuracy:", accuracy_score(y_val, val_pred))
        print("Test Accuracy:", accuracy_score(y_test, test_pred))
        print(
            classification_report(
                y_test,
                test_pred,
                target_names=label_encoder.classes_,
                zero_division=0
            )
        )


# -------------------------
# MAIN
# -------------------------

def main():
    df = prepare_dataframe()
    df = build_audio_files(df)

    X_audio, labels, latitudes, longitudes = build_embeddings(df)

    train_and_evaluate(X_audio, labels, latitudes, longitudes)


if __name__ == "__main__":
    main()