import os
import hashlib
import random
import warnings
import contextlib
import requests
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score


# -------------------------
# CONFIG
# -------------------------

CSV_PATH = "birds.csv"
AUDIO_DIR = "audio_files"

SAMPLE_RATE = 22050
DURATION = 5
NUM_SAMPLES = SAMPLE_RATE * DURATION

BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-3

NUM_GENERA = 8
MAX_PER_CLASS = 200
MIN_PER_CLASS = 60

RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

warnings.filterwarnings("ignore")


# -------------------------
# SUPPRESS AUDIO DECODER SPAM
# -------------------------

@contextlib.contextmanager
def suppress_stderr():
    """
    Suppresses low-level C/C++ stderr output too,
    including mpg123/torchaudio decoder spam.
    """
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stderr_fd = os.dup(2)

    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(old_stderr_fd, 2)
        os.close(old_stderr_fd)
        os.close(devnull_fd)


def safe_torchaudio_load(path):
    with suppress_stderr():
        return torchaudio.load(path)


# -------------------------
# DOWNLOAD AUDIO
# -------------------------

def url_to_filename(url):
    hashed = hashlib.md5(str(url).encode()).hexdigest()
    return f"{hashed}.mp3"


def download_audio(url, save_path):
    """
    Downloads and verifies audio.
    Bad/corrupted/non-audio files are deleted and skipped.
    """

    # If file already exists, verify it first
    if os.path.exists(save_path):
        try:
            if os.path.getsize(save_path) > 1000:
                safe_torchaudio_load(save_path)
                return True
            else:
                os.remove(save_path)
        except Exception:
            try:
                os.remove(save_path)
            except Exception:
                pass

    try:
        response = requests.get(url, timeout=20, allow_redirects=True)

        if response.status_code != 200:
            return False

        content_type = response.headers.get("Content-Type", "").lower()

        # Avoid saving HTML pages as fake mp3 files
        if "text/html" in content_type:
            return False

        with open(save_path, "wb") as f:
            f.write(response.content)

        # Skip files that are too tiny to be real audio
        if os.path.getsize(save_path) < 1000:
            os.remove(save_path)
            return False

        # Verify torchaudio can open it
        try:
            safe_torchaudio_load(save_path)
        except Exception:
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


# -------------------------
# DATASET
# -------------------------

class BirdSoundDataset(Dataset):
    def __init__(self, dataframe, label_encoder, augment=False):
        self.df = dataframe.reset_index(drop=True)
        self.label_encoder = label_encoder
        self.augment = augment

        self.mel_transform = MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=1024,
            hop_length=512,
            n_mels=96
        )

        self.db_transform = AmplitudeToDB()

    def __len__(self):
        return len(self.df)

    def load_audio(self, path):
        waveform, sr = safe_torchaudio_load(path)

        # Stereo to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)

        # Pad or crop to fixed length
        if waveform.shape[1] < NUM_SAMPLES:
            pad_amount = NUM_SAMPLES - waveform.shape[1]
            waveform = F.pad(waveform, (0, pad_amount))
        else:
            if self.augment:
                max_start = waveform.shape[1] - NUM_SAMPLES
                start = random.randint(0, max_start)
                waveform = waveform[:, start:start + NUM_SAMPLES]
            else:
                waveform = waveform[:, :NUM_SAMPLES]

        return waveform

    def augment_waveform(self, waveform):
        # Add small noise
        if random.random() < 0.35:
            waveform = waveform + torch.randn_like(waveform) * 0.004

        # Random volume shift
        if random.random() < 0.35:
            gain = random.uniform(0.85, 1.15)
            waveform = waveform * gain

        return waveform

    def augment_spectrogram(self, spec):
        # Frequency mask
        if random.random() < 0.45:
            freq_bins = spec.shape[1]
            mask_size = random.randint(4, 12)
            start = random.randint(0, max(0, freq_bins - mask_size))
            spec[:, start:start + mask_size, :] = spec.mean()

        # Time mask
        if random.random() < 0.45:
            time_bins = spec.shape[2]
            mask_size = random.randint(8, 24)
            start = random.randint(0, max(0, time_bins - mask_size))
            spec[:, :, start:start + mask_size] = spec.mean()

        return spec

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        audio_path = row["audio_path"]
        label = row["label"]

        try:
            waveform = self.load_audio(audio_path)
        except Exception:
            waveform = torch.zeros((1, NUM_SAMPLES))

        if self.augment:
            waveform = self.augment_waveform(waveform)

        mel = self.mel_transform(waveform)
        mel_db = self.db_transform(mel)

        # Normalize per example
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

        if self.augment:
            mel_db = self.augment_spectrogram(mel_db)

        y = self.label_encoder.transform([label])[0]
        y = torch.tensor(y, dtype=torch.long)

        return mel_db, y


# -------------------------
# CNN MODEL
# -------------------------

class BirdCNN(nn.Module):
    def __init__(self, num_classes):
        super(BirdCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.20),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.40),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# -------------------------
# TRAINING FUNCTIONS
# -------------------------

def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()

    total_loss = 0
    all_preds = []
    all_labels = []

    for X, y in dataloader:
        X = X.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        logits = model(X)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / max(1, len(dataloader))
    acc = accuracy_score(all_labels, all_preds)

    return avg_loss, acc


def evaluate(model, dataloader, criterion):
    model.eval()

    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(X)
            loss = criterion(logits, y)

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / max(1, len(dataloader))
    acc = accuracy_score(all_labels, all_preds)

    return avg_loss, acc, all_labels, all_preds


def make_class_weights(labels, label_encoder):
    encoded = label_encoder.transform(labels)
    counts = pd.Series(encoded).value_counts().sort_index()

    weights = []
    total = len(encoded)
    num_classes = len(label_encoder.classes_)

    for class_index in range(num_classes):
        count = counts.get(class_index, 1)
        weight = total / (num_classes * count)
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


# -------------------------
# MAIN
# -------------------------

def main():
    os.makedirs(AUDIO_DIR, exist_ok=True)

    print("Using device:", DEVICE)

    df = pd.read_csv(CSV_PATH)

    print("\nCSV columns:")
    print(df.columns.tolist())

    if "sound_url" not in df.columns:
        raise ValueError("CSV must have a 'sound_url' column.")

    # -------------------------
    # Create label column
    # -------------------------

    if "scientific_name" in df.columns:
        df = df.dropna(subset=["sound_url", "scientific_name"])
        df["genus"] = df["scientific_name"].astype(str).str.split().str[0]
        df["label"] = df["genus"].astype(str)
        label_type = "genus"
    elif "taxon_id" in df.columns:
        df = df.dropna(subset=["sound_url", "taxon_id"])
        df["label"] = df["taxon_id"].astype(str)
        label_type = "taxon_id"
    else:
        raise ValueError("CSV must have either 'scientific_name' or 'taxon_id'.")

    df = df[df["sound_url"].notna()].copy()
    df = df[df["sound_url"].astype(str).str.startswith("http")].copy()

    # -------------------------
    # Keep top classes and limit files per class
    # -------------------------

    top_labels = df["label"].value_counts().head(NUM_GENERA).index
    df = df[df["label"].isin(top_labels)].copy()

    df = (
        df.groupby("label", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), MAX_PER_CLASS), random_state=RANDOM_SEED))
        .reset_index(drop=True)
    )

    print(f"\nUsing label type: {label_type}")
    print(f"\nTop {NUM_GENERA} labels after capping at {MAX_PER_CLASS} each:")
    print(df["label"].value_counts())

    # -------------------------
    # Download and verify audio
    # -------------------------

    audio_paths = []
    keep_rows = []

    print("\nDownloading and verifying audio files...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        url = str(row["sound_url"])
        filename = url_to_filename(url)
        path = os.path.join(AUDIO_DIR, filename)

        success = download_audio(url, path)

        if success and os.path.exists(path):
            audio_paths.append(path)
            keep_rows.append(idx)

    df = df.loc[keep_rows].copy()
    df["audio_path"] = audio_paths

    print(f"\nUsable audio files: {len(df)}")

    if len(df) == 0:
        raise ValueError("No usable audio files were downloaded. Check your sound_url column.")

    print("\nLabel counts after audio verification:")
    print(df["label"].value_counts())

    # Remove classes with too few usable files
    valid_labels = df["label"].value_counts()
    valid_labels = valid_labels[valid_labels >= MIN_PER_CLASS].index
    df = df[df["label"].isin(valid_labels)].copy()

    actual_num_classes = df["label"].nunique()

    if actual_num_classes < 2:
        raise ValueError("Need at least 2 classes with enough usable audio files.")

    print(f"\nClasses kept after filtering minimum {MIN_PER_CLASS} files/class: {actual_num_classes}")
    print(df["label"].value_counts())

    # -------------------------
    # Encode labels
    # -------------------------

    label_encoder = LabelEncoder()
    label_encoder.fit(df["label"])

    print("\nClass mapping:")
    for i, label in enumerate(label_encoder.classes_):
        print(i, "=", label)

    # -------------------------
    # Train / val / test split
    # -------------------------

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=df["label"]
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_df["label"]
    )

    print("\nSplit sizes:")
    print("Train:", len(train_df))
    print("Val:", len(val_df))
    print("Test:", len(test_df))

    train_dataset = BirdSoundDataset(train_df, label_encoder, augment=True)
    val_dataset = BirdSoundDataset(val_df, label_encoder, augment=False)
    test_dataset = BirdSoundDataset(test_df, label_encoder, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # -------------------------
    # Model setup
    # -------------------------

    model = BirdCNN(num_classes=actual_num_classes).to(DEVICE)

    class_weights = make_class_weights(train_df["label"], label_encoder)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    # -------------------------
    # Training
    # -------------------------

    print("\nTraining model...")

    best_val_acc = 0
    best_val_loss = float("inf")
    best_model_path = "bird_genus_cnn_best.pt"

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_acc, _, _ = evaluate(
            model,
            val_loader,
            criterion
        )

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": label_encoder.classes_,
                    "label_type": label_type,
                    "sample_rate": SAMPLE_RATE,
                    "duration": DURATION,
                    "num_classes": actual_num_classes
                },
                best_model_path
            )

            print("Saved new best model.")

    # -------------------------
    # Final test evaluation
    # -------------------------

    print("\nLoading best model for final test evaluation...")

    checkpoint = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, y_true, y_pred = evaluate(
        model,
        test_loader,
        criterion
    )

    print("\nFinal Test Loss:", test_loss)
    print("Final Test Accuracy:", test_acc)

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[str(c) for c in label_encoder.classes_],
            zero_division=0
        )
    )

    print(f"\nBest model saved as {best_model_path}")


if __name__ == "__main__":
    main()