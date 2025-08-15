import os
import time
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from prepare_data import build_data_map, extract_feature_from_image_bgr, IMG_SIZE

MODEL_PATH = "siamese_ml_torch.pth"
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ==================== Siamese-like MLP ==================== #
class SiameseMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

# ==================== Tạo feature pair ==================== #
def make_pair_feature(e1, e2):
    e1 = np.asarray(e1).ravel()
    e2 = np.asarray(e2).ravel()
    diff = np.abs(e1 - e2)
    return np.concatenate([e1, e2, diff]).astype(np.float32)

def build_pairs_from_embeddings(embeddings, labels, class_to_idxs, neg_per_pos=1):
    X_pairs, y_pairs = [], []
    # Positive
    for lab, idxs in class_to_idxs.items():
        L = len(idxs)
        if L < 2:
            continue
        for i in range(L):
            for j in range(i+1, L):
                X_pairs.append(make_pair_feature(embeddings[idxs[i]], embeddings[idxs[j]]))
                y_pairs.append(1)
    # Negative
    num_pos = sum(y_pairs)
    num_neg_target = num_pos * neg_per_pos
    neg_count = 0
    all_idxs = np.arange(len(labels))
    while neg_count < num_neg_target:
        i, j = np.random.choice(all_idxs, 2, replace=False)
        if labels[i] != labels[j]:
            X_pairs.append(make_pair_feature(embeddings[i], embeddings[j]))
            y_pairs.append(0)
            neg_count += 1
    return np.vstack(X_pairs).astype(np.float32), np.array(y_pairs).astype(np.int64)

# ==================== Train model ==================== #
def train_siamese_ml_torch(dataset_path, pca_dim=128, batch_size=32, epochs=20, lr=1e-3, neg_per_pos=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # --- Load dataset ---
    data_map = build_data_map(dataset_path)
    persons = list(data_map.keys())
    features_raw, labels, class_to_idxs = [], [], {}
    idx_counter = 0
    for person_idx, person in enumerate(persons):
        for img_path in data_map[person]:
            img = cv2.imread(img_path)
            if img is None:
                continue
            feat = extract_feature_from_image_bgr(img)
            if feat is None:
                continue
            features_raw.append(feat)
            labels.append(person_idx)
            class_to_idxs.setdefault(person_idx, []).append(idx_counter)
            idx_counter += 1

    X_raw = np.vstack(features_raw)
    y = np.array(labels)

    # --- Preprocessing ---
    selector = VarianceThreshold(1e-5)
    X_sel = selector.fit_transform(X_raw)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sel)
    pca = PCA(n_components=min(pca_dim, X_scaled.shape[1]), whiten=True, random_state=RANDOM_SEED)
    X_emb = pca.fit_transform(X_scaled)

    # --- Build pairs ---
    X_pairs, y_pairs = build_pairs_from_embeddings(X_emb, y, class_to_idxs, neg_per_pos)
    X_train, X_val, y_train, y_val = train_test_split(X_pairs, y_pairs, test_size=0.2, stratify=y_pairs, random_state=RANDOM_SEED)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # --- Model ---
    input_dim = X_pairs.shape[1]
    model = SiameseMLP(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --- Training loop ---
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                preds = (torch.sigmoid(outputs) > 0.5).int()
                correct += (preds == yb.int()).sum().item()
                total += yb.size(0)
        val_acc = correct / total
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Val Acc: {val_acc:.4f} - Time: {elapsed:.2f}s")

    # --- Build gallery embeddings ---
    gallery_embeddings = []
    gallery_names = []
    for person in sorted(set(persons)):
        idxs = [i for i, lab in enumerate(labels) if lab == persons.index(person)]
        emb_mean = np.mean(X_emb[idxs], axis=0)
        gallery_embeddings.append(emb_mean)
        gallery_names.append(person)
    gallery_embeddings = np.vstack(gallery_embeddings)

    # --- Save model ---
    torch.save({
        "model_state": model.state_dict(),
        "selector": selector,
        "scaler": scaler,
        "pca": pca,
        "persons": persons,
        "gallery_embeddings": gallery_embeddings,
        "gallery_names": gallery_names
    }, MODEL_PATH)
    print(f"✅ Model trained và lưu tại {MODEL_PATH}")

# ==================== Build gallery từ dataset mới ==================== #
def build_gallery_from_new_dataset(dataset_path):
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    selector = checkpoint["selector"]
    scaler = checkpoint["scaler"]
    pca = checkpoint["pca"]
    model_state = checkpoint["model_state"]
    persons = checkpoint["persons"]

    data_map = build_data_map(dataset_path)
    features_raw, names = [], []
    for person in data_map:
        for img_path in data_map[person]:
            img = cv2.imread(img_path)
            if img is None:
                continue
            feat = extract_feature_from_image_bgr(img)
            if feat is None:
                continue
            features_raw.append(feat)
            names.append(person)

    X_raw = np.vstack(features_raw)
    X_sel = selector.transform(X_raw)
    X_scaled = scaler.transform(X_sel)
    X_emb = pca.transform(X_scaled)

    gallery_embeddings = []
    gallery_names = []
    for person in sorted(set(names)):
        idxs = [i for i, n in enumerate(names) if n == person]
        emb_mean = np.mean(X_emb[idxs], axis=0)
        gallery_embeddings.append(emb_mean)
        gallery_names.append(person)

    checkpoint["gallery_embeddings"] = np.vstack(gallery_embeddings)
    checkpoint["gallery_names"] = gallery_names
    torch.save(checkpoint, MODEL_PATH)
    print(f"✅ Đã cập nhật gallery từ {len(gallery_names)} người")

if __name__ == "__main__":
    DATASET_PATH = r"D:\AI\project\dataset"
    NEW_DATASET_PATH = r"D:\AI\project\dataset_update"

    train_siamese_ml_torch(DATASET_PATH)
    build_gallery_from_new_dataset(NEW_DATASET_PATH)

1
