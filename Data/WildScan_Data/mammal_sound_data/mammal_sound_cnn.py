"""
mammal_sound_general_hybrid.py

Generalizable mammal sound classifier using:
1. CNN audio-only model on mel-spectrograms
2. Metadata-only Random Forest model
3. Strong hybrid model:
   CNN audio embeddings + handcrafted audio features + geographic/time metadata
4. Soft voting ensemble

Recommended generalizable setting:
- Use all species with at least MIN_SOUNDS_PER_SPECIES clips
- Do not downsample classes by default
- Use class weights instead
- Use longer audio duration

Expected CSV columns:
- id
- sound_url
- common_name
- quality_grade
- latitude
- longitude
- observed_on
- time_observed_at
- positional_accuracy

Run:
    python mammal_sound_general_hybrid.py

Install:
    pip install pandas numpy requests tqdm librosa scikit-learn tensorflow matplotlib joblib
"""

import os
import requests
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    VotingClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif


# =========================
# SETTINGS
# =========================

CSV_PATH = "mammal_sounds.csv"
AUDIO_DIR = "mammal_audio"

LABEL_COL = "common_name"

SAMPLE_RATE = 22050

# Longer audio gives the model more chance to hear the actual call.
DURATION = 10

N_MELS = 128
RANDOM_STATE = 42

# Recommended generalizable setting:
# Use all species with enough clips instead of only top N.
TOP_N_SPECIES = None
MIN_SOUNDS_PER_SPECIES = 80

# Leave this as None for generalization.
# Use this only for a controlled high-accuracy experiment.
SELECTED_SPECIES = None

# For generalization, do NOT downsample by default.
# This keeps more data and uses class weights instead.
BALANCE_CLASSES = False
MAX_PER_CLASS = None

EPOCHS = 35
BATCH_SIZE = 8


# =========================
# LOAD AND FILTER DATA
# =========================

def load_and_filter_data(csv_path):
    df = pd.read_csv(csv_path)

    print("Original rows:", len(df))
    print("Columns:", list(df.columns))

    required_cols = ["id", "sound_url", LABEL_COL]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.dropna(subset=["sound_url", LABEL_COL]).copy()
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()

    print("\nSpecies/classes before filtering:", df[LABEL_COL].nunique())

    if "quality_grade" in df.columns:
        df = df[df["quality_grade"] == "research"].copy()

    print("Species/classes after research-grade filtering:", df[LABEL_COL].nunique())

    if SELECTED_SPECIES is not None:
        df = df[df[LABEL_COL].isin(SELECTED_SPECIES)].copy()
        print("\nUsing manually selected species:")
        print(SELECTED_SPECIES)

    elif TOP_N_SPECIES is not None:
        top_species = df[LABEL_COL].value_counts().head(TOP_N_SPECIES).index
        df = df[df[LABEL_COL].isin(top_species)].copy()
        print(f"\nUsing top {TOP_N_SPECIES} most frequent species.")

    else:
        species_counts = df[LABEL_COL].value_counts()
        valid_species = species_counts[
            species_counts >= MIN_SOUNDS_PER_SPECIES
        ].index

        df = df[df[LABEL_COL].isin(valid_species)].copy()
        print(f"\nUsing all species with at least {MIN_SOUNDS_PER_SPECIES} sounds.")

    print("\nAfter species filtering:")
    print("Rows:", len(df))
    print("Species/classes:", df[LABEL_COL].nunique())
    print(df[LABEL_COL].value_counts())

    if df[LABEL_COL].nunique() < 2:
        raise ValueError("Need at least 2 species/classes to train.")

    return df


# =========================
# DOWNLOAD AUDIO
# =========================

def download_audio_file(audio_id, url):
    os.makedirs(AUDIO_DIR, exist_ok=True)

    filepath = os.path.join(AUDIO_DIR, f"{audio_id}.wav")

    if os.path.exists(filepath):
        return filepath

    try:
        response = requests.get(url, timeout=20)

        if response.status_code == 200:
            with open(filepath, "wb") as f:
                f.write(response.content)
            return filepath

        return None

    except Exception:
        return None


def download_all_audio(df):
    filepaths = []

    print("\nDownloading audio files...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        filepath = download_audio_file(row["id"], row["sound_url"])
        filepaths.append(filepath)

    df["filepath"] = filepaths
    df = df.dropna(subset=["filepath"]).copy()

    print("\nSuccessfully downloaded/found:", len(df))

    print("\nClass counts after downloading:")
    print(df[LABEL_COL].value_counts())

    return df


# =========================
# BALANCE CLASSES
# =========================

def balance_classes(df, label_col, max_per_class=None):
    counts = df[label_col].value_counts()

    if max_per_class is None:
        max_per_class = counts.min()

    balanced_parts = []

    for class_name in counts.index:
        class_df = df[df[label_col] == class_name]

        n = min(len(class_df), max_per_class)

        sampled = class_df.sample(
            n=n,
            random_state=RANDOM_STATE
        )

        balanced_parts.append(sampled)

    balanced_df = pd.concat(balanced_parts)

    balanced_df = balanced_df.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    print("\nBalanced class counts:")
    print(balanced_df[label_col].value_counts())

    return balanced_df


# =========================
# METADATA FEATURES
# =========================

def build_metadata_features(df):
    metadata = pd.DataFrame(index=df.index)

    metadata["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce"
    ) if "latitude" in df.columns else np.nan

    metadata["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce"
    ) if "longitude" in df.columns else np.nan

    metadata["positional_accuracy"] = pd.to_numeric(
        df["positional_accuracy"],
        errors="coerce"
    ) if "positional_accuracy" in df.columns else np.nan

    if "observed_on" in df.columns:
        observed_on = pd.to_datetime(df["observed_on"], errors="coerce")
        metadata["month"] = observed_on.dt.month
        metadata["day_of_year"] = observed_on.dt.dayofyear
    else:
        metadata["month"] = np.nan
        metadata["day_of_year"] = np.nan

    if "time_observed_at" in df.columns:
        time_observed = pd.to_datetime(df["time_observed_at"], errors="coerce")
        metadata["hour"] = time_observed.dt.hour
    else:
        metadata["hour"] = np.nan

    # Cyclical time features
    metadata["month_sin"] = np.sin(2 * np.pi * metadata["month"] / 12)
    metadata["month_cos"] = np.cos(2 * np.pi * metadata["month"] / 12)

    metadata["day_sin"] = np.sin(2 * np.pi * metadata["day_of_year"] / 365)
    metadata["day_cos"] = np.cos(2 * np.pi * metadata["day_of_year"] / 365)

    metadata["hour_sin"] = np.sin(2 * np.pi * metadata["hour"] / 24)
    metadata["hour_cos"] = np.cos(2 * np.pi * metadata["hour"] / 24)

    metadata = metadata.drop(columns=["month", "day_of_year", "hour"])

    return metadata


# =========================
# AUDIO FEATURES
# =========================

def load_fixed_audio(filepath):
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE, duration=DURATION)

    target_length = SAMPLE_RATE * DURATION

    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]

    return y, sr


def audio_to_mel(filepath):
    try:
        y, sr = load_fixed_audio(filepath)

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=N_MELS,
            n_fft=2048,
            hop_length=512
        )

        mel_db = librosa.power_to_db(mel, ref=np.max)

        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

        return mel_db

    except Exception:
        return None


def extract_audio_features(filepath):
    try:
        y, sr = load_fixed_audio(filepath)

        features = []

        # MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))
        features.extend(np.max(mfcc, axis=1))
        features.extend(np.min(mfcc, axis=1))

        # Delta MFCCs
        mfcc_delta = librosa.feature.delta(mfcc)
        features.extend(np.mean(mfcc_delta, axis=1))
        features.extend(np.std(mfcc_delta, axis=1))

        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.extend(np.mean(chroma, axis=1))
        features.extend(np.std(chroma, axis=1))

        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features.extend(np.mean(contrast, axis=1))
        features.extend(np.std(contrast, axis=1))

        # Tonnetz
        try:
            harmonic_y = librosa.effects.harmonic(y)
            tonnetz = librosa.feature.tonnetz(y=harmonic_y, sr=sr)
            features.extend(np.mean(tonnetz, axis=1))
            features.extend(np.std(tonnetz, axis=1))
        except Exception:
            features.extend([0.0] * 12)

        # Other audio descriptors
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)

        for arr in [
            spectral_centroid,
            spectral_bandwidth,
            spectral_rolloff,
            zero_crossing_rate,
            rms
        ]:
            features.append(np.mean(arr))
            features.append(np.std(arr))
            features.append(np.max(arr))
            features.append(np.min(arr))

        return np.array(features, dtype=np.float32)

    except Exception:
        return None


def build_audio_dataset(df):
    X_audio = []
    X_audio_features = []
    valid_indices = []

    print("\nConverting audio to mel-spectrograms and handcrafted audio features...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        mel = audio_to_mel(row["filepath"])
        audio_features = extract_audio_features(row["filepath"])

        if mel is not None and audio_features is not None:
            X_audio.append(mel)
            X_audio_features.append(audio_features)
            valid_indices.append(idx)

    X_audio = np.array(X_audio)
    X_audio = X_audio[..., np.newaxis]

    X_audio_features = np.array(X_audio_features)

    df_valid = df.loc[valid_indices].copy()

    print("\nAudio dataset ready:")
    print("X_audio shape:", X_audio.shape)
    print("X_audio_features shape:", X_audio_features.shape)

    return X_audio, X_audio_features, df_valid


# =========================
# CNN AUDIO MODEL
# =========================

def build_cnn_feature_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.30)(x)

    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.35)(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.40)(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    embedding = tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        name="audio_embedding"
    )(x)

    x = tf.keras.layers.Dropout(0.60)(embedding)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    embedding_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer("audio_embedding").output
    )

    return model, embedding_model


# =========================
# PLOTS
# =========================

def plot_training_history(history):
    plt.figure()
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CNN Training vs Validation Accuracy")
    plt.legend()
    plt.savefig("cnn_accuracy_plot.png")
    plt.close()

    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN Training vs Validation Loss")
    plt.legend()
    plt.savefig("cnn_loss_plot.png")
    plt.close()


# =========================
# EVALUATION HELPER
# =========================

def evaluate_model(name, y_true, y_pred, label_encoder):
    print("\n==============================")
    print(name)
    print("==============================")

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("Accuracy:", acc)
    print("Macro F1:", macro_f1)
    print("Weighted F1:", weighted_f1)

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=label_encoder.classes_,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    return acc, macro_f1, weighted_f1


# =========================
# MAIN
# =========================

def main():
    df = load_and_filter_data(CSV_PATH)

    df = download_all_audio(df)

    if BALANCE_CLASSES:
        df = balance_classes(df, LABEL_COL, max_per_class=MAX_PER_CLASS)
    else:
        print("\nNot downsampling classes. Using all available data with class weights.")

    X_audio, X_audio_features, df_valid = build_audio_dataset(df)

    metadata = build_metadata_features(df_valid)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_valid[LABEL_COL])

    print("\nFinal classes:")
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"{i}: {class_name}")

    print("\nFinal class counts after audio processing:")
    print(pd.Series(label_encoder.inverse_transform(y)).value_counts())

    print("\nFinal dataset shapes:")
    print("Audio:", X_audio.shape)
    print("Handcrafted audio features:", X_audio_features.shape)
    print("Metadata:", metadata.shape)
    print("Labels:", y.shape)

    indices = np.arange(len(y))

    train_idx, test_idx, y_train, y_test = train_test_split(
        indices,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    train_idx, val_idx, y_train, y_val = train_test_split(
        train_idx,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train
    )

    X_audio_train = X_audio[train_idx]
    X_audio_val = X_audio[val_idx]
    X_audio_test = X_audio[test_idx]

    X_audio_features_train = X_audio_features[train_idx]
    X_audio_features_val = X_audio_features[val_idx]
    X_audio_features_test = X_audio_features[test_idx]

    X_meta_train = metadata.iloc[train_idx]
    X_meta_val = metadata.iloc[val_idx]
    X_meta_test = metadata.iloc[test_idx]

    print("\nSplit sizes:")
    print("Train:", len(train_idx))
    print("Validation:", len(val_idx))
    print("Test:", len(test_idx))

    print("\nTrain class counts:")
    print(pd.Series(y_train).value_counts().sort_index())

    print("\nValidation class counts:")
    print(pd.Series(y_val).value_counts().sort_index())

    print("\nTest class counts:")
    print(pd.Series(y_test).value_counts().sort_index())

    # =========================
    # MODEL 1: CNN AUDIO-ONLY
    # =========================

    cnn_model, embedding_model = build_cnn_feature_model(
        input_shape=X_audio_train.shape[1:],
        num_classes=len(label_encoder.classes_)
    )

    cnn_model.summary()

    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )

    class_weights = {
        int(i): float(weight)
        for i, weight in enumerate(class_weights_array)
    }

    print("\nClass weights:")
    print(class_weights)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            mode="max",
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    ]

    print("\nTraining CNN audio-only model...")

    history = cnn_model.fit(
        X_audio_train,
        y_train,
        validation_data=(X_audio_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weights
    )

    cnn_probs = cnn_model.predict(X_audio_test)
    cnn_preds = np.argmax(cnn_probs, axis=1)

    cnn_acc, cnn_macro_f1, cnn_weighted_f1 = evaluate_model(
        "MODEL 1: CNN Audio-Only",
        y_test,
        cnn_preds,
        label_encoder
    )

    plot_training_history(history)

    # =========================
    # MODEL 2: METADATA-ONLY
    # =========================

    print("\nTraining metadata-only Random Forest model...")

    metadata_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=1200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])

    metadata_model.fit(X_meta_train, y_train)

    meta_preds = metadata_model.predict(X_meta_test)

    meta_acc, meta_macro_f1, meta_weighted_f1 = evaluate_model(
        "MODEL 2: Metadata-Only Random Forest",
        y_test,
        meta_preds,
        label_encoder
    )

    # =========================
    # MODEL 3: STRONG HYBRID
    # =========================

    print("\nExtracting CNN audio embeddings...")

    audio_embed_train = embedding_model.predict(X_audio_train)
    audio_embed_val = embedding_model.predict(X_audio_val)
    audio_embed_test = embedding_model.predict(X_audio_test)

    print("Audio embedding train shape:", audio_embed_train.shape)
    print("Handcrafted audio feature train shape:", X_audio_features_train.shape)

    meta_preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    meta_train_processed = meta_preprocessor.fit_transform(X_meta_train)
    meta_val_processed = meta_preprocessor.transform(X_meta_val)
    meta_test_processed = meta_preprocessor.transform(X_meta_test)

    audio_feature_preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    audio_features_train_processed = audio_feature_preprocessor.fit_transform(
        X_audio_features_train
    )
    audio_features_val_processed = audio_feature_preprocessor.transform(
        X_audio_features_val
    )
    audio_features_test_processed = audio_feature_preprocessor.transform(
        X_audio_features_test
    )

    X_hybrid_train = np.hstack([
        audio_embed_train,
        audio_features_train_processed,
        meta_train_processed
    ])

    X_hybrid_val = np.hstack([
        audio_embed_val,
        audio_features_val_processed,
        meta_val_processed
    ])

    X_hybrid_test = np.hstack([
        audio_embed_test,
        audio_features_test_processed,
        meta_test_processed
    ])

    print("\nHybrid feature shapes:")
    print("Train:", X_hybrid_train.shape)
    print("Validation:", X_hybrid_val.shape)
    print("Test:", X_hybrid_test.shape)

    # More features are okay now because we are keeping more data.
    k_features = min(150, X_hybrid_train.shape[1])

    feature_selector = SelectKBest(
        score_func=f_classif,
        k=k_features
    )

    X_hybrid_train_selected = feature_selector.fit_transform(
        X_hybrid_train,
        y_train
    )
    X_hybrid_val_selected = feature_selector.transform(X_hybrid_val)
    X_hybrid_test_selected = feature_selector.transform(X_hybrid_test)

    candidate_models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=1800,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=1800,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

        "SVM RBF": SVC(
            C=3.0,
            gamma="scale",
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE
        ),

        "Logistic Regression": LogisticRegression(
            C=2.0,
            max_iter=5000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
    }

    best_name = None
    best_model = None
    best_val_macro_f1 = -1

    print("\nTuning hybrid models on validation set...")

    for name, clf in candidate_models.items():
        clf.fit(X_hybrid_train_selected, y_train)

        val_preds = clf.predict(X_hybrid_val_selected)
        val_acc = accuracy_score(y_val, val_preds)
        val_macro_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)

        print(f"{name} validation accuracy: {val_acc:.4f}")
        print(f"{name} validation macro F1: {val_macro_f1:.4f}")

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_name = name
            best_model = clf

    print("\nBest hybrid model:", best_name)
    print("Best validation macro F1:", best_val_macro_f1)

    X_hybrid_train_full = np.vstack([
        X_hybrid_train_selected,
        X_hybrid_val_selected
    ])

    y_train_full = np.concatenate([y_train, y_val])

    final_hybrid_model = best_model
    final_hybrid_model.fit(X_hybrid_train_full, y_train_full)

    hybrid_preds = final_hybrid_model.predict(X_hybrid_test_selected)

    hybrid_acc, hybrid_macro_f1, hybrid_weighted_f1 = evaluate_model(
        f"MODEL 3: Strong Hybrid Audio + Metadata ({best_name})",
        y_test,
        hybrid_preds,
        label_encoder
    )

    # =========================
    # MODEL 4: SOFT VOTING ENSEMBLE
    # =========================

    print("\nTraining soft voting hybrid ensemble...")

    ensemble_models = [
        ("rf", RandomForestClassifier(
            n_estimators=1200,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )),
        ("extra", ExtraTreesClassifier(
            n_estimators=1200,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )),
        ("svm", SVC(
            C=3.0,
            gamma="scale",
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE
        ))
    ]

    voting_model = VotingClassifier(
        estimators=ensemble_models,
        voting="soft"
    )

    voting_model.fit(X_hybrid_train_full, y_train_full)

    voting_preds = voting_model.predict(X_hybrid_test_selected)

    voting_acc, voting_macro_f1, voting_weighted_f1 = evaluate_model(
        "MODEL 4: Soft Voting Hybrid Ensemble",
        y_test,
        voting_preds,
        label_encoder
    )

    # =========================
    # SUMMARY
    # =========================

    results = {
        "CNN Audio-Only": (cnn_acc, cnn_macro_f1),
        "Metadata-Only": (meta_acc, meta_macro_f1),
        f"Strong Hybrid ({best_name})": (hybrid_acc, hybrid_macro_f1),
        "Soft Voting Hybrid Ensemble": (voting_acc, voting_macro_f1)
    }

    print("\n==============================")
    print("FINAL SUMMARY")
    print("==============================")

    for model_name, (acc, macro_f1) in results.items():
        print(f"{model_name} Accuracy: {acc:.4f}")
        print(f"{model_name} Macro F1: {macro_f1:.4f}")

    best_report_name = max(results, key=lambda name: results[name][1])

    print("\nBest model by Macro F1:", best_report_name)
    print("Best model Accuracy:", results[best_report_name][0])
    print("Best model Macro F1:", results[best_report_name][1])

    # Save everything
    cnn_model.save("cnn_audio_model.keras")
    embedding_model.save("cnn_audio_embedding_model.keras")

    joblib.dump(metadata_model, "metadata_random_forest.joblib")
    joblib.dump(meta_preprocessor, "metadata_preprocessor.joblib")
    joblib.dump(audio_feature_preprocessor, "audio_feature_preprocessor.joblib")
    joblib.dump(feature_selector, "hybrid_feature_selector.joblib")
    joblib.dump(final_hybrid_model, "strong_hybrid_audio_metadata_model.joblib")
    joblib.dump(voting_model, "soft_voting_hybrid_ensemble.joblib")

    np.save("label_classes_common_names.npy", label_encoder.classes_)

    print("\nSaved:")
    print("- cnn_audio_model.keras")
    print("- cnn_audio_embedding_model.keras")
    print("- metadata_random_forest.joblib")
    print("- metadata_preprocessor.joblib")
    print("- audio_feature_preprocessor.joblib")
    print("- hybrid_feature_selector.joblib")
    print("- strong_hybrid_audio_metadata_model.joblib")
    print("- soft_voting_hybrid_ensemble.joblib")
    print("- label_classes_common_names.npy")


if __name__ == "__main__":
    main()