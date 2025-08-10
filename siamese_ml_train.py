# siamese_ml_train.py
import os
import random
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from prepare_data import build_data_map, extract_feature_from_image_bgr, IMG_SIZE

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def build_single_features(dataset_path):
    data_map = build_data_map(dataset_path)
    persons = list(data_map.keys())
    features_raw = []
    labels = []
    img_paths = []
    person_to_idxs = {p: [] for p in persons}

    for person_idx, person in enumerate(persons):
        for pth in data_map[person]:
            img = extract_img_cv2(pth)
            if img is None:
                continue
            feat = extract_feature_from_image_bgr(img)
            if feat is None:
                continue
            idx = len(features_raw)
            features_raw.append(feat)
            labels.append(person_idx)
            img_paths.append(pth)
            person_to_idxs[person].append(idx)

    return features_raw, labels, img_paths, persons, person_to_idxs


def extract_img_cv2(path):
    try:
        import cv2
        return cv2.imread(path)
    except Exception:
        return None


def make_pair_feature(e1, e2):
    e1 = np.asarray(e1).ravel()
    e2 = np.asarray(e2).ravel()
    diff = np.abs(e1 - e2)
    return np.concatenate([e1, e2, diff]).astype(np.float32)


def build_pairs_from_embeddings(embeddings, labels, person_to_idxs, neg_per_pos=1):
    N = len(labels)
    X_pairs = []
    y_pairs = []

    # Positive pairs
    for person_idx, idxs in person_to_idxs.items() if isinstance(list(person_to_idxs.keys())[0], str) else []:
        pass  # not used, we will use integer-key mapping below

    # Convert person_to_idxs to int-key dict if needed
    # here assume person_to_idxs original keys are person names; we will rely on labels for class indices
    # Build mapping class_idx -> idxs
    class_to_idxs = {}
    for i, lab in enumerate(labels):
        class_to_idxs.setdefault(lab, []).append(i)

    # positives
    for lab, idxs in class_to_idxs.items():
        L = len(idxs)
        if L < 2:
            continue
        # all combinations (could be many); limit to reasonable number per person
        for i in range(L):
            for j in range(i + 1, L):
                e1 = embeddings[idxs[i]]
                e2 = embeddings[idxs[j]]
                X_pairs.append(make_pair_feature(e1, e2))
                y_pairs.append(1)

    # negatives: sample random pairs from different classes
    # target negative count = neg_per_pos * num_pos
    num_pos = sum(1 for v in y_pairs if v == 1)
    num_neg_target = num_pos * neg_per_pos
    neg_count = 0
    all_idxs = np.arange(len(labels))
    while neg_count < num_neg_target:
        i, j = np.random.choice(all_idxs, size=2, replace=False)
        if labels[i] != labels[j]:
            X_pairs.append(make_pair_feature(embeddings[i], embeddings[j]))
            y_pairs.append(0)
            neg_count += 1

    X_pairs = np.vstack(X_pairs).astype(np.float32)
    y_pairs = np.array(y_pairs).astype(np.int32)
    return X_pairs, y_pairs


def train_siamese_ml(dataset_path,
                     out_model_path="siamese_ml_model.pkl",
                     pca_dim=128,
                     neg_per_pos=1,
                     test_size=0.15):
    # 1) extract raw features for each image
    data_map = build_data_map(dataset_path)
    persons = list(data_map.keys())
    total_people = len(persons)
    total_images = sum(len(v) for v in data_map.values())
    print(f"People: {total_people}, Images: {total_images}")

    features_raw = []
    labels = []
    img_paths = []
    class_to_idxs = {}

    # load images & extract raw features (same as earlier code but simpler)
    idx_counter = 0
    for person_idx, person in enumerate(persons):
        for pth in data_map[person]:
            img = extract_img_cv2(pth)
            if img is None:
                continue
            feat = extract_feature_from_image_bgr(img)
            if feat is None:
                continue
            features_raw.append(feat)
            labels.append(person_idx)
            img_paths.append(pth)
            class_to_idxs.setdefault(person_idx, []).append(idx_counter)
            idx_counter += 1

    if len(features_raw) == 0:
        raise RuntimeError("No valid images found.")

    X_raw = np.vstack(features_raw)  # shape (N, D_raw)
    y = np.array(labels)

    # 2) Fit single-image pipeline: VarianceThreshold -> StandardScaler -> PCA
    selector = VarianceThreshold(threshold=1e-5)
    X_sel = selector.fit_transform(X_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sel)

    pca = PCA(n_components=min(pca_dim, X_scaled.shape[1]), whiten=True, random_state=RANDOM_SEED)
    X_emb = pca.fit_transform(X_scaled)  # embeddings for each single image

    # 3) Build pairs from embeddings
    X_pairs, y_pairs = build_pairs_from_embeddings(X_emb, y.tolist(), class_to_idxs, neg_per_pos=neg_per_pos)
    print(f"Built pairs: {X_pairs.shape}, labels: {y_pairs.shape}; positive ratio: {y_pairs.mean():.3f}")

    # 4) Train MLPClassifier on pairs
    X_tr, X_val, y_tr, y_val = train_test_split(X_pairs, y_pairs, test_size=test_size, random_state=RANDOM_SEED, stratify=y_pairs)
    clf = MLPClassifier(hidden_layer_sizes=(512, 256), activation='relu', solver='adam',
                        max_iter=400, random_state=RANDOM_SEED, early_stopping=True, verbose=False)
    clf.fit(X_tr, y_tr)
    train_acc = clf.score(X_tr, y_tr)
    val_acc = clf.score(X_val, y_val)
    print(f"MLP train acc: {train_acc:.4f}, val acc: {val_acc:.4f}")

    # 5) Build gallery: mean embedding per person (in embedding space)
    gallery_embeddings = []
    gallery_names = []
    for person_idx, person in enumerate(persons):
        idxs = class_to_idxs.get(person_idx, [])
        if len(idxs) == 0:
            continue
        emb_person = X_emb[idxs]
        emb_mean = np.mean(emb_person, axis=0)
        gallery_embeddings.append(emb_mean)
        gallery_names.append(person)

    gallery_embeddings = np.vstack(gallery_embeddings)
    # 6) Save everything
    saved = {
        "selector": selector,
        "scaler": scaler,
        "pca": pca,
        "pair_clf": clf,
        "gallery_embeddings": gallery_embeddings,
        "gallery_names": gallery_names,
        "img_size": IMG_SIZE
    }
    joblib.dump(saved, out_model_path)
    print(f"Saved siamese-ml model to: {out_model_path}")


if __name__ == "__main__":
    DATASET_PATH = r"D:\AI\project\dataset_update"  # change to your dataset
    train_siamese_ml(DATASET_PATH, out_model_path="siamese_ml_model.pkl",
                     pca_dim=128, neg_per_pos=1, test_size=0.15)

