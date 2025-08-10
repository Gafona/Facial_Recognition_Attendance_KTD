import os
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

IMG_SIZE = (128, 128)
HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
    "transform_sqrt": True
}
LBP_P = 8
LBP_R = 1
LBP_METHOD = "uniform"
LBP_BINS = LBP_P + 2

def extract_feature_from_image_bgr(img_bgr):
    if img_bgr is None:
        return None

    img = cv2.resize(img_bgr, IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ==== Tiền xử lý ảnh ====
    gray = cv2.equalizeHist(gray)          # cân bằng sáng
    gray = cv2.GaussianBlur(gray, (3, 3), 0) # giảm nhiễu nhẹ

    # ==== HOG ====
    hog_vec = hog(gray,
                  orientations=HOG_PARAMS["orientations"],
                  pixels_per_cell=HOG_PARAMS["pixels_per_cell"],
                  cells_per_block=HOG_PARAMS["cells_per_block"],
                  block_norm=HOG_PARAMS["block_norm"],
                  transform_sqrt=HOG_PARAMS["transform_sqrt"],
                  feature_vector=True)

    # ==== LBP ====
    lbp = local_binary_pattern(gray, P=LBP_P, R=LBP_R, method=LBP_METHOD)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_BINS + 1), range=(0, LBP_BINS))
    hist = hist.astype("float32")
    if hist.sum() > 0:
        hist /= hist.sum()

    # Không chuẩn hóa tại đây, sẽ chuẩn hóa sau trong pipeline train
    feat = np.hstack([hog_vec.astype("float32"), hist])
    return feat

def build_data_map(dataset_path):
    data = {}
    for person in sorted(os.listdir(dataset_path)):
        p = os.path.join(dataset_path, person)
        if os.path.isdir(p):
            imgs = [os.path.join(p, f) for f in sorted(os.listdir(p))
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
            if imgs:
                data[person] = imgs
    return data
