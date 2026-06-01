import os
import requests
import pandas as pd
from PIL import Image
from io import BytesIO

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# =========================
# SETTINGS
# =========================

CSV_FILE = "observations.csv"
IMAGE_DIR = "images"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LR = 0.0003

MAX_IMAGES_PER_GENUS = 300
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# LOAD + CLEAN DATA
# =========================

df = pd.read_csv(CSV_FILE, encoding="utf-8", engine="python")

df = df.dropna(subset=["image_url", "taxon_genus_name"])
df = df.drop_duplicates(subset=["image_url"])

# balance dataset
df = (
    df.groupby("taxon_genus_name", group_keys=False)
      .apply(lambda x: x.sample(n=min(len(x), MAX_IMAGES_PER_GENUS), random_state=42))
      .reset_index(drop=True)
)

print("Total images:", len(df))
print("Genera:", df["taxon_genus_name"].nunique())


# =========================
# DOWNLOAD IMAGES
# =========================

os.makedirs(IMAGE_DIR, exist_ok=True)

def download_image(row):
    genus = str(row["taxon_genus_name"]).replace(" ", "_")
    img_id = str(row["id"])
    path = os.path.join(IMAGE_DIR, f"{genus}_{img_id}.jpg")

    if os.path.exists(path):
        return path

    try:
        r = requests.get(row["image_url"], timeout=10)
        if r.status_code == 200:
            img = Image.open(BytesIO(r.content)).convert("RGB")
            img.save(path)
            return path
    except:
        return None

    return None


paths = []
for _, row in df.iterrows():
    paths.append(download_image(row))

df["image_path"] = paths
df = df.dropna(subset=["image_path"]).reset_index(drop=True)

print("Downloaded images:", len(df))


# =========================
# LABELS
# =========================

le = LabelEncoder()
df["label"] = le.fit_transform(df["taxon_genus_name"])

num_classes = len(le.classes_)
print("Classes:", le.classes_)


# =========================
# DATASET
# =========================

class PlantDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = Image.open(row["image_path"]).convert("RGB")
        img = self.transform(img)

        label = torch.tensor(row["label"], dtype=torch.long)

        return img, label


train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = PlantDataset(train_df, train_transform)
val_dataset = PlantDataset(val_df, val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# =========================
# MODEL
# =========================

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, num_classes)
)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# =========================
# TRAIN
# =========================

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total

    # validation
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            outputs = model(imgs)
            preds = outputs.argmax(1)

            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss {total_loss:.3f} | Train {train_acc:.3f} | Val {val_acc:.3f}")


# =========================
# SAVE
# =========================

torch.save(model.state_dict(), "genus_model.pth")

pd.DataFrame({
    "label": range(len(le.classes_)),
    "genus": le.classes_
}).to_csv("labels.csv", index=False)

print("done.")