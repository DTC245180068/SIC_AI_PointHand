# -*- coding: utf-8 -*-
import os, glob, math, time, threading
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "gesture_cnn.pt")
LANDMARK_CACHE_PATH = os.path.join(BASE_DIR, "landmark_dataset.npz")
YOLO_DATASET_DIR = os.path.join(BASE_DIR, "Hand Gesture.v1i.yolov8")
MP_MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")

PARENT_IDS = np.array(
    [0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19],
    dtype=np.int32,
)
TIP_IDS = np.array([4, 8, 12, 16, 20], dtype=np.int32)
DIP_IDS = np.array([3, 7, 11, 15, 19], dtype=np.int32)
PIP_IDS = np.array([3, 6, 10, 14, 18], dtype=np.int32)
MCP_IDS = np.array([2, 5, 9, 13, 17], dtype=np.int32)

IN_CHANNELS = 6
NUM_CLASSES = 7

GESTURE_CLASSES = {
    0: "0: NAM TAY",
    1: "1: BUT DO (Ctrl+P)",
    2: "2: CHIEU (Shift+F5) / TO HL",
    3: "3: DA QUANG (Ctrl+I)",
    4: "4: TAY XOA (Ctrl+E)",
    5: "5: THOAT BUT (Ctrl+A)",
    6: "6: DEN LASER (Pinch OK - Ctrl+L)",
}


class GestureCNN(nn.Module):
    """Mạng 1D-CNN (Residual + Dual Pooling) phân loại 7 cử chỉ từ 21 khớp tay đẳng hướng."""

    def __init__(self, num_classes: int = NUM_CLASSES, in_channels: int = IN_CHANNELS):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 48, 3, padding=1, bias=False),
            nn.BatchNorm1d(48),
            nn.GELU(),
        )
        self.res = nn.Sequential(
            nn.Conv1d(48, 48, 3, padding=1, bias=False),
            nn.BatchNorm1d(48),
            nn.GELU(),
            nn.Conv1d(48, 48, 3, padding=1, bias=False),
            nn.BatchNorm1d(48),
        )
        self.enc = nn.Sequential(
            nn.Conv1d(48, 96, 3, padding=1, bias=False),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.Conv1d(96, 128, 3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(256, 96),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(96, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = F.gelu(h + self.res(h))
        h = self.enc(h)
        pooled = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=1)
        return self.head(pooled)


def _landmarks_to_iso(lms_or_pts, aspect: float = 1.0) -> np.ndarray:
    if isinstance(lms_or_pts, np.ndarray):
        pts = np.array(lms_or_pts, dtype=np.float32, copy=True)
    else:
        pts = np.fromiter((c for lm in lms_or_pts for c in (lm.x, lm.y)), dtype=np.float32, count=42).reshape(21, 2)
    if aspect != 1.0:
        pts[:, 0] *= aspect
    return pts


def extract_cnn_features(lms_or_pts, label: str = "Right", aspect: float = 1.0) -> np.ndarray:
    """Trích xuất 6 kênh đặc trưng bất biến tịnh tiến/tỉ lệ/xoay từ 21 khớp tay đẳng hướng 1:1."""
    pts = _landmarks_to_iso(lms_or_pts, aspect=aspect)
    rel = pts - pts[0]
    if label == "Left":
        rel[:, 0] = -rel[:, 0]
    palm_scale = max(float(np.linalg.norm(rel[9])), 1e-4)
    rel /= palm_scale

    # Xoay sao cho trục cổ tay (0) -> MCP ngón giữa (9) hướng lên (-Y)
    a = -math.atan2(float(rel[9, 0]), -float(rel[9, 1]))
    ca, sa = math.cos(a), math.sin(a)
    rx = rel[:, 0] * ca - rel[:, 1] * sa
    ry = rel[:, 0] * sa + rel[:, 1] * ca

    d_wrist = np.hypot(rx, ry)
    d_thumb = np.hypot(rx - rx[4], ry - ry[4])
    dx_par = rx - rx[PARENT_IDS]
    dy_par = ry - ry[PARENT_IDS]
    return np.stack([rx, ry, d_wrist, d_thumb, dx_par, dy_par], axis=0).astype(np.float32)


def fast_finger_states_eval(pts_iso: np.ndarray) -> np.ndarray:
    """Vector hóa nhận diện 5 ngón tay đẳng hướng 1:1 bằng khoảng cách & góc 2 khớp (PIP + DIP)."""
    wrist, tips, dips, pips, mcps = pts_iso[0], pts_iso[TIP_IDS], pts_iso[DIP_IDS], pts_iso[PIP_IDS], pts_iso[MCP_IDS]
    d2_tip = np.sum((tips[1:] - wrist) ** 2, axis=1)
    d2_pip = np.sum((pips[1:] - wrist) ** 2, axis=1)
    d2_mcp = np.sum((mcps[1:] - wrist) ** 2, axis=1)

    v1 = pips[1:] - mcps[1:]
    v2 = dips[1:] - pips[1:]
    v3 = tips[1:] - dips[1:]
    n1, n2, n3 = np.linalg.norm(v1, axis=1), np.linalg.norm(v2, axis=1), np.linalg.norm(v3, axis=1)
    cos_pip = np.sum(v1 * v2, axis=1) / np.maximum(1e-6, n1 * n2)
    cos_dip = np.sum(v2 * v3, axis=1) / np.maximum(1e-6, n2 * n3)

    up = np.empty(5, dtype=bool)
    up[1:] = (
        (d2_tip > d2_pip * 1.08)
        & (d2_tip > d2_mcp * 1.10)
        & (cos_pip > 0.25)
        & (cos_dip > 0.18)
    )

    vt1, vt2 = pts_iso[3] - pts_iso[2], pts_iso[4] - pts_iso[3]
    cos_t = float(np.dot(vt1, vt2)) / max(1e-6, float(np.linalg.norm(vt1) * np.linalg.norm(vt2)))
    d2_p_tip = float(np.sum((pts_iso[4] - pts_iso[17]) ** 2))
    d2_p_ip  = float(np.sum((pts_iso[3] - pts_iso[17]) ** 2))
    d2_m_tip = float(np.sum((pts_iso[4] - pts_iso[9]) ** 2))
    d2_m_ip  = float(np.sum((pts_iso[2] - pts_iso[9]) ** 2))
    up[0] = ((d2_p_tip > d2_p_ip * 1.08) or (d2_m_tip > d2_m_ip * 1.12)) and (cos_t > 0.20)
    return up


def _sample_pose(cls_id: int, mask, rng: np.random.Generator, pinch_variant: str = "") -> np.ndarray:
    """Sinh 21 điểm khớp tay đẳng hướng cho đủ 7 lớp 0..6."""
    pts = np.zeros((21, 2), dtype=np.float32)
    pts[1] = [-0.28 + rng.normal(0, 0.02), -0.24 + rng.normal(0, 0.02)]
    pts[2] = [-0.46 + rng.normal(0, 0.03), -0.48 + rng.normal(0, 0.02)]

    if cls_id == 6:
        # Cử chỉ Chụm ngón cái & trỏ (Pinch OK 👌 / Pinch kín 🤏)
        tx = -0.32 + rng.normal(0, 0.04)
        ty = -1.04 + rng.normal(0, 0.05)
        pts[3] = [-0.46 + rng.normal(0, 0.03), -0.78 + rng.normal(0, 0.03)]
        pts[4] = [tx, ty]
        if pinch_variant == "closed":
            four_up = [0, 0, 0, 0]
        elif pinch_variant == "1up":
            four_up = [0, 1, 0, 0]
        elif pinch_variant == "2up":
            four_up = [0, 1, 1, 0]
        else:
            four_up = [0, 1, 1, 1]
    else:
        if mask[0]:
            pts[3] = [-0.66 + rng.normal(0, 0.04), -0.68 + rng.normal(0, 0.04)]
            pts[4] = [-0.88 + rng.normal(0, 0.05), -0.88 + rng.normal(0, 0.05)]
        else:
            pts[3] = [-0.30 + rng.normal(0, 0.04), -0.56 + rng.normal(0, 0.04)]
            pts[4] = [-0.14 + rng.normal(0, 0.05), -0.60 + rng.normal(0, 0.05)]
        four_up = mask[1:]

    mcps = [(-0.30, -0.92, 0.95), (-0.09, -1.00, 1.06), (0.12, -0.95, 0.96), (0.31, -0.84, 0.78)]
    for i, (mcp, pip, dip, tip) in enumerate([(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]):
        bx, by, L = mcps[i]
        L *= rng.uniform(0.92, 1.08)
        pts[mcp] = [bx, by]
        if cls_id == 6 and i == 0:
            # Ngón trỏ uốn vòng chạm vào đầu ngón cái
            pts[pip] = [bx + rng.uniform(-0.04, 0.06), by - 0.24]
            pts[dip] = [pts[4, 0] + rng.uniform(-0.04, 0.06), pts[4, 1] - rng.uniform(0.05, 0.12)]
            pts[tip] = [pts[4, 0] + rng.normal(0, 0.035), pts[4, 1] + rng.normal(0, 0.035)]
            continue

        sp = (i - 1.5) * rng.uniform(0.04, 0.11)
        if four_up[i]:
            pts[pip] = [bx + sp * 0.35, by - L * 0.38]
            pts[dip] = [bx + sp * 0.70, by - L * 0.69]
            pts[tip] = [bx + sp * 1.05, by - L * rng.uniform(0.92, 1.06)]
        else:
            pts[pip] = [bx + rng.normal(0, 0.02), by - L * rng.uniform(0.18, 0.30)]
            pts[dip] = [bx + rng.normal(0, 0.03), by + L * rng.uniform(0.00, 0.10)]
            pts[tip] = [bx + rng.normal(0, 0.03), by + L * rng.uniform(0.08, 0.22)]

    pts[1:] += rng.normal(0.0, 0.020, size=(20, 2)).astype(np.float32)
    pts[:, 0] *= rng.uniform(0.86, 1.14)
    pts[:, 1] *= rng.uniform(0.88, 1.14)
    a = rng.uniform(-0.78, 0.78)
    ca, sa = math.cos(a), math.sin(a)
    rot = np.stack([pts[:, 0] * ca - pts[:, 1] * sa, pts[:, 0] * sa + pts[:, 1] * ca], axis=1)
    return rot.astype(np.float32)


def _augment_pts(pts_iso: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Tăng cường dữ liệu trực tiếp trên 21 khớp thật (xoay ±60°, co giãn, lật trái/phải, nhiễu)."""
    pts = np.array(pts_iso, dtype=np.float32, copy=True)
    pts -= pts[0]
    if rng.random() < 0.5:
        pts[:, 0] = -pts[:, 0]
    pts[:, 0] *= rng.uniform(0.86, 1.14)
    pts[:, 1] *= rng.uniform(0.86, 1.14)
    a = rng.uniform(-1.05, 1.05)  # ±60 độ
    ca, sa = math.cos(a), math.sin(a)
    rx = pts[:, 0] * ca - pts[:, 1] * sa
    ry = pts[:, 0] * sa + pts[:, 1] * ca
    out = np.stack([rx, ry], axis=1).astype(np.float32)
    palm_scale = max(0.05, float(np.linalg.norm(out[9])))
    out[1:] += rng.normal(0.0, 0.016 * palm_scale, size=(20, 2)).astype(np.float32)
    return out


def clean_raw_landmarks(X_in: np.ndarray, y_in: np.ndarray):
    """
    Lọc bỏ các mẫu nhãn sai rõ rệt trong Roboflow (như 5 ngón xòe gán nhãn 1/3/4),
    và chuẩn hóa các cử chỉ hợp lệ (chuyển [1,1,1,0,0] về Lớp 3 Dạ quang).
    """
    cleaned_X, cleaned_y = [], []
    stats = {"kept": 0, "reassigned": 0, "rejected": 0, "by_class_in": {}, "by_class_out": {}}

    for pts, orig_c in zip(X_in, y_in):
        c_int = int(orig_c)
        stats["by_class_in"][c_int] = stats["by_class_in"].get(c_int, 0) + 1
        s = fast_finger_states_eval(pts)
        n_four = int(np.sum(s[1:]))
        n_all = int(np.sum(s))

        actual_c = None
        if n_all == 5 or (n_four == 4 and s[0]):
            actual_c = 5
        elif n_four == 4 or n_all == 4:
            actual_c = 4
        elif (n_four == 3 and not s[0]) or (n_all == 3 and s[0] and s[1] and s[2]):
            actual_c = 3
        elif n_four == 2 and s[1] and s[2] and not s[0]:
            actual_c = 2
        elif n_four == 1 and s[1]:
            actual_c = 1
        elif n_four == 0:
            actual_c = 0

        if actual_c is None:
            stats["rejected"] += 1
            continue

        if actual_c == c_int:
            cleaned_X.append(pts)
            cleaned_y.append(actual_c)
            stats["kept"] += 1
            stats["by_class_out"][actual_c] = stats["by_class_out"].get(actual_c, 0) + 1
        elif c_int == 2 and actual_c == 3:
            # Thumb + Index + Middle chuẩn hóa về Lớp 3 (3 ngón)
            cleaned_X.append(pts)
            cleaned_y.append(3)
            stats["reassigned"] += 1
            stats["by_class_out"][3] = stats["by_class_out"].get(3, 0) + 1
        elif c_int == 5 and actual_c == 4:
            # 4 ngón không xòe ngón cái chuẩn hóa về Lớp 4
            cleaned_X.append(pts)
            cleaned_y.append(4)
            stats["reassigned"] += 1
            stats["by_class_out"][4] = stats["by_class_out"].get(4, 0) + 1
        else:
            # Mâu thuẫn hoàn toàn (nhãn bẩn) -> loại bỏ
            stats["rejected"] += 1

    return (
        np.array(cleaned_X, dtype=np.float32) if cleaned_X else np.zeros((0, 21, 2), dtype=np.float32),
        np.array(cleaned_y, dtype=np.int64) if cleaned_y else np.zeros((0,), dtype=np.int64),
        stats,
    )


def refine_and_balance_dataset(
    target_train_per_class: int = 1200,
    target_valid_per_class: int = 150,
    target_test_per_class: int = 100,
    cache_path: str = LANDMARK_CACHE_PATH,
    verbose: bool = True,
):
    """
    Tinh chỉnh toàn bộ dữ liệu:
    1. Lọc sạch nhãn sai trong Roboflow (Train/Valid/Test).
    2. Cân bằng Lớp 0 (Nắm tay) và Lớp 6 (Laser Pinch) cho cả 3 tập.
    3. Lưu bộ dữ liệu chuẩn 7 lớp vào landmark_dataset.npz.
    """
    rng = np.random.default_rng(2026)

    # Đọc cache thô nếu có, hoặc trích xuất từ ảnh
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        X_tr_raw, y_tr_raw = data["X_train"], data["y_train"]
        X_va_raw, y_va_raw = data["X_valid"], data["y_valid"]
        X_te_raw, y_te_raw = data["X_test"], data["y_test"]
    else:
        cached = extract_yolo_landmarks(verbose=verbose)
        if cached is None:
            raise RuntimeError(f"Khong tim thay du lieu anh tai {YOLO_DATASET_DIR}")
        X_tr_raw, y_tr_raw, X_va_raw, y_va_raw, X_te_raw, y_te_raw = cached

    if verbose:
        print("\n" + "=" * 65)
        print(" BƯỚC 1: TINH CHỈNH & LỌC NHÃN SAI BỘ DỮ LIỆU ROBOFLOW ")
        print("=" * 65)

    X_tr_clean, y_tr_clean, st_tr = clean_raw_landmarks(X_tr_raw, y_tr_raw)
    X_va_clean, y_va_clean, st_va = clean_raw_landmarks(X_va_raw, y_va_raw)
    X_te_clean, y_te_clean, st_te = clean_raw_landmarks(X_te_raw, y_te_raw)

    if verbose:
        print(f"[*] Train : {len(y_tr_raw)} mau goc -> {len(y_tr_clean)} mau sach (loai {st_tr['rejected']} nhan sai, chuan hoa {st_tr['reassigned']} mau)")
        print(f"[*] Valid : {len(y_va_raw)} mau goc -> {len(y_va_clean)} mau sach (loai {st_va['rejected']} nhan sai, chuan hoa {st_va['reassigned']} mau)")
        print(f"[*] Test  : {len(y_te_raw)} mau goc -> {len(y_te_clean)} mau sach (loai {st_te['rejected']} nhan sai, chuan hoa {st_te['reassigned']} mau)")

    # -------------------------------------------------------------
    # BƯỚC 2: CÂN BẰNG & BỔ SUNG 7 LỚP (0..6) CHO TẬP TRAIN
    # -------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 65)
        print(f" BƯỚC 2: CÂN BẰNG TẬP HUẤN LUYỆN 7 LỚP ({target_train_per_class} MẪU/LỚP) ")
        print("=" * 65)

    pinch_variants = ["ok_3up", "ok_3up", "ok_2up", "ok_1up", "closed"]
    patterns = {
        0: [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0]],
        1: [[0, 1, 0, 0, 0], [1, 1, 0, 0, 0]],
        2: [[0, 1, 1, 0, 0]],
        3: [[0, 1, 1, 1, 0], [1, 1, 1, 0, 0]],
        4: [[0, 1, 1, 1, 1], [1, 1, 1, 1, 0], [1, 1, 1, 0, 1], [1, 1, 0, 1, 1], [1, 0, 1, 1, 1]],
        5: [[1, 1, 1, 1, 1]],
    }

    clean_pool_tr = {c: [] for c in range(NUM_CLASSES)}
    for pts, c in zip(X_tr_clean, y_tr_clean):
        clean_pool_tr[int(c)].append(pts)

    refined_X_tr, refined_y_tr = [], []
    for c in range(NUM_CLASSES):
        pool = clean_pool_tr[c]
        if c == 6:
            # Lớp 6: Laser Pinch 👌 đa biến thể
            for _ in range(target_train_per_class):
                pv = pinch_variants[int(rng.integers(0, len(pinch_variants)))]
                refined_X_tr.append(_sample_pose(6, None, rng, pinch_variant=pv))
                refined_y_tr.append(6)
        elif c == 0:
            # Lớp 0: Nắm tay (oversample 38 mẫu thật + bổ sung đa góc)
            for i in range(target_train_per_class):
                if pool and (i < len(pool) * 12):
                    base = pool[i % len(pool)]
                    refined_X_tr.append(_augment_pts(base, rng))
                else:
                    m = patterns[0][int(rng.integers(0, len(patterns[0])))]
                    refined_X_tr.append(_sample_pose(0, m, rng))
                refined_y_tr.append(0)
        else:
            # Lớp 1..5: Ưu tiên tối đa mẫu thật sạch + augment
            n_real_use = min(len(pool), target_train_per_class)
            for i in range(n_real_use):
                refined_X_tr.append(_augment_pts(pool[i], rng) if rng.random() < 0.65 else pool[i])
                refined_y_tr.append(c)
            # Thiếu thì bù mẫu tổng hợp chuẩn
            n_remain = target_train_per_class - n_real_use
            for i in range(n_remain):
                if pool:
                    base = pool[i % len(pool)]
                    refined_X_tr.append(_augment_pts(base, rng))
                else:
                    m = patterns[c][int(rng.integers(0, len(patterns[c])))]
                    refined_X_tr.append(_sample_pose(c, m, rng))
                refined_y_tr.append(c)

    refined_X_tr = np.stack(refined_X_tr).astype(np.float32)
    refined_y_tr = np.array(refined_y_tr, dtype=np.int64)

    # -------------------------------------------------------------
    # BƯỚC 3: CÂN BẰNG TẬP VALIDATION & TEST CHUẨN 7 LỚP
    # -------------------------------------------------------------
    def balance_eval_split(clean_pool, target_per_c):
        out_X, out_y = [], []
        for c in range(NUM_CLASSES):
            pool = clean_pool.get(c, [])
            if c == 6:
                for _ in range(target_per_c):
                    pv = pinch_variants[int(rng.integers(0, len(pinch_variants)))]
                    out_X.append(_sample_pose(6, None, rng, pinch_variant=pv))
                    out_y.append(6)
            elif c == 0:
                for i in range(target_per_c):
                    if pool:
                        out_X.append(_augment_pts(pool[i % len(pool)], rng))
                    else:
                        m = patterns[0][int(rng.integers(0, len(patterns[0])))]
                        out_X.append(_sample_pose(0, m, rng))
                    out_y.append(0)
            else:
                n_real = min(len(pool), target_per_c)
                for i in range(n_real):
                    out_X.append(pool[i])
                    out_y.append(c)
                for i in range(target_per_c - n_real):
                    base = pool[i % len(pool)] if pool else _sample_pose(c, patterns[c][0], rng)
                    out_X.append(_augment_pts(base, rng))
                    out_y.append(c)
        return np.stack(out_X).astype(np.float32), np.array(out_y, dtype=np.int64)

    clean_pool_va = {c: [] for c in range(NUM_CLASSES)}
    for pts, c in zip(X_va_clean, y_va_clean):
        clean_pool_va[int(c)].append(pts)

    clean_pool_te = {c: [] for c in range(NUM_CLASSES)}
    for pts, c in zip(X_te_clean, y_te_clean):
        clean_pool_te[int(c)].append(pts)

    refined_X_va, refined_y_va = balance_eval_split(clean_pool_va, target_valid_per_class)
    refined_X_te, refined_y_te = balance_eval_split(clean_pool_te, target_test_per_class)

    # Lưu bộ dữ liệu chuẩn 7 lớp vào cache
    np.savez_compressed(
        cache_path,
        X_train=refined_X_tr, y_train=refined_y_tr,
        X_valid=refined_X_va, y_valid=refined_y_va,
        X_test=refined_X_te,   y_test=refined_y_te,
        X_test_raw=X_te_raw,  y_test_raw=y_te_raw,
    )

    if verbose:
        print(f"[OK] Luu bo du lieu tinh chinh 7 lop vao {cache_path} ({os.path.getsize(cache_path) / 1024:.1f} KB)")
        print(f"     Train : {refined_X_tr.shape[0]} mau (7 lop x {target_train_per_class})")
        print(f"     Valid : {refined_X_va.shape[0]} mau (7 lop x {target_valid_per_class})")
        print(f"     Test  : {refined_X_te.shape[0]} mau (7 lop x {target_test_per_class})")
        print(f"     Luu kem Test goc: {len(y_te_raw)} mau de so sanh lich su.")

    return (
        refined_X_tr, refined_y_tr,
        refined_X_va, refined_y_va,
        refined_X_te, refined_y_te,
        X_te_raw, y_te_raw,
    )


def extract_yolo_landmarks(
    dataset_dir: str = YOLO_DATASET_DIR,
    cache_path: str = LANDMARK_CACHE_PATH,
    workers: int = 4,
    verbose: bool = True,
):
    """Trích xuất tọa độ 21 khớp MediaPipe từ tập ảnh Hand Gesture.v1i.yolov8."""
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        if "X_test_raw" in data.files:
            return (
                data["X_train"], data["y_train"],
                data["X_valid"], data["y_valid"],
                data["X_test"], data["y_test"],
                data["X_test_raw"], data["y_test_raw"],
            )
        return (
            data["X_train"], data["y_train"],
            data["X_valid"], data["y_valid"],
            data["X_test"], data["y_test"],
            data["X_test"], data["y_test"],
        )

    if not os.path.isdir(dataset_dir) or not os.path.exists(MP_MODEL_PATH):
        return None

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    tls = threading.local()

    def get_detector():
        det = getattr(tls, "detector", None)
        if det is None:
            det = mp_vision.HandLandmarker.create_from_options(
                mp_vision.HandLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=MP_MODEL_PATH),
                    running_mode=mp_vision.RunningMode.IMAGE,
                    num_hands=1,
                    min_hand_detection_confidence=0.40,
                    min_hand_presence_confidence=0.40,
                )
            )
            tls.detector = det
        return det

    def process_item(item):
        img_path, cls_id = item
        img = cv2.imread(img_path)
        if img is None:
            return None
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = get_detector().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not res.hand_landmarks:
            return None
        lms = res.hand_landmarks[0]
        label = "Right" if not res.handedness else res.handedness[0][0].category_name
        pts_iso = np.array([[lm.x * (w / max(1, h)), lm.y] for lm in lms], dtype=np.float32)
        if label == "Left":
            pts_iso[:, 0] = -pts_iso[:, 0]
        return pts_iso, int(cls_id)

    splits = {}
    t0 = time.time()
    for split in ("train", "valid", "test"):
        lbl_files = sorted(glob.glob(os.path.join(dataset_dir, split, "labels", "*.txt")))
        items = []
        for lp in lbl_files:
            with open(lp, "r", encoding="utf-8", errors="ignore") as f:
                lines = [l.strip().split() for l in f if l.strip()]
            if len(lines) != 1:
                continue
            cls_id = int(lines[0][0])
            if not (0 <= cls_id <= 5):
                continue
            stem = os.path.splitext(os.path.basename(lp))[0]
            for ext in (".jpg", ".jpeg", ".png", ".webp"):
                ip = os.path.join(dataset_dir, split, "images", stem + ext)
                if os.path.exists(ip):
                    items.append((ip, cls_id))
                    break

        pts_list, y_list = [], []
        if verbose:
            print(f"[Data] Trich xuat 21 khop MediaPipe cho '{split}' ({len(items)} anh)...")
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for out in ex.map(process_item, items):
                if out is not None:
                    pts_list.append(out[0])
                    y_list.append(out[1])
        splits[split] = (
            np.stack(pts_list, axis=0).astype(np.float32) if pts_list else np.zeros((0, 21, 2), dtype=np.float32),
            np.array(y_list, dtype=np.int64),
        )

    return refine_and_balance_dataset(cache_path=cache_path, verbose=verbose)


def _evaluate_loader(model: nn.Module, X_feat: np.ndarray, y_true: np.ndarray, num_classes: int = NUM_CLASSES):
    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(X_feat))
        preds = logits.argmax(dim=1).numpy()
    acc = float((preds == y_true).mean()) if len(y_true) else 0.0
    per_cls = {}
    f1s = []
    for c in range(num_classes):
        tp = int(((preds == c) & (y_true == c)).sum())
        fp = int(((preds == c) & (y_true != c)).sum())
        fn = int(((preds != c) & (y_true == c)).sum())
        n_c = tp + fn
        if n_c > 0:
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            f1 = 2 * prec * rec / max(1e-8, prec + rec)
            per_cls[c] = (tp, n_c, rec, f1)
            f1s.append(f1)
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    return acc, macro_f1, per_cls, preds


def build_and_train_cnn(save_path: str = MODEL_SAVE_PATH, epochs: int = 35, verbose: bool = True) -> GestureCNN:
    """Huấn luyện GestureCNN trên toàn bộ tập dữ liệu đã được tinh chỉnh & cân bằng chuẩn 7 lớp."""
    data_tuple = refine_and_balance_dataset(verbose=verbose)
    X_tr_pts, y_train, X_va_pts, y_valid, X_te_pts, y_test, X_te_raw, y_te_raw = data_tuple

    if verbose:
        print("\n" + "=" * 65)
        print(" BƯỚC 3: TRÍCH XUẤT ĐẶC TRƯNG 6 KÊNH BẤT BIẾN CHO GESTURECNN ")
        print("=" * 65)

    X_train = np.stack([extract_cnn_features(p, label="Right", aspect=1.0) for p in X_tr_pts]).astype(np.float32)
    X_valid = np.stack([extract_cnn_features(p, label="Right", aspect=1.0) for p in X_va_pts]).astype(np.float32)
    X_test  = np.stack([extract_cnn_features(p, label="Right", aspect=1.0) for p in X_te_pts]).astype(np.float32)

    loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=64, shuffle=True)

    torch.manual_seed(42)
    model = GestureCNN(NUM_CLASSES, IN_CHANNELS)
    opt = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=2e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)
    crit = nn.CrossEntropyLoss(label_smoothing=0.03)

    best_score = -1.0
    best_state = None
    t_train = time.time()

    if verbose:
        print("\n" + "=" * 65)
        print(f" BƯỚC 4: TIẾN HÀNH HUẤN LUYỆN GESTURECNN ({epochs} EPOCHS) ")
        print("=" * 65)

    for ep in range(1, epochs + 1):
        model.train()
        tot_loss = 0.0
        for bx, by in loader:
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            opt.step()
            tot_loss += float(loss.item()) * len(by)
        sched.step()

        val_acc, val_f1, _, _ = _evaluate_loader(model, X_valid, y_valid)
        score = 0.5 * val_acc + 0.5 * val_f1
        if score >= best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if verbose and (ep == 1 or ep % 5 == 0 or ep == epochs):
            print(
                f"Epoch {ep:02d}/{epochs} | loss={tot_loss / len(y_train):.4f} "
                f"| val_acc={val_acc * 100:.2f}% | val_macro_f1={val_f1 * 100:.2f}%"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    torch.save(model.state_dict(), save_path)

    if verbose:
        print(f"\n[OK] Luu trong so moi vao {save_path} (hoan tat trong {time.time() - t_train:.1f}s)")
        print("\n" + "=" * 65)
        print(" BƯỚC 5: ĐÁNH GIÁ CHẤT LƯỢNG MÔ HÌNH SAU TINH CHỈNH DỮ LIỆU ")
        print("=" * 65)

        te_acc, te_f1, te_cls, _ = _evaluate_loader(model, X_test, y_test)
        print(f"=== [1] TẬP TEST 7 LỚP TINH CHỈNH (n={len(y_test)}): Acc={te_acc*100:.2f}%, Macro-F1={te_f1*100:.2f}% ===")
        for c, (tp, n_c, rec, f1) in sorted(te_cls.items()):
            print(f"  Class {c} ({GESTURE_CLASSES[c]}): {tp}/{n_c} ({rec*100:.1f}%), F1={f1*100:.1f}%")

        if len(X_te_raw) > 0:
            X_real_raw = np.stack([extract_cnn_features(p, label="Right", aspect=1.0) for p in X_te_raw]).astype(np.float32)
            raw_acc, raw_f1, raw_cls, _ = _evaluate_loader(model, X_real_raw, y_te_raw)
            print(f"\n=== [2] TẬP ROBOFLOW TEST GỐC (n={len(y_te_raw)}): Acc={raw_acc*100:.2f}%, Macro-F1={raw_f1*100:.2f}% ===")
            for c, (tp, n_c, rec, f1) in sorted(raw_cls.items()):
                print(f"  Class {c} ({GESTURE_CLASSES[c]}): {tp}/{n_c} ({rec*100:.1f}%), F1={f1*100:.1f}%")

        # Đo độ trễ suy luận CPU thực tế trên 1 frame
        sample = torch.from_numpy(X_test[:1])
        with torch.inference_mode():
            for _ in range(50):
                model(sample)
            t_inf = time.perf_counter()
            for _ in range(1000):
                model(sample)
            dt_ms = (time.perf_counter() - t_inf)
        print(f"\n[Latency] PyTorch CPU forward (1 frame): {dt_ms:.3f} ms/frame")

    return model


def load_or_train_cnn() -> GestureCNN:
    model = GestureCNN(NUM_CLASSES, IN_CHANNELS)
    if os.path.exists(MODEL_SAVE_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location="cpu", weights_only=True))
            model.eval()
            return model
        except Exception:
            pass
    return build_and_train_cnn(MODEL_SAVE_PATH, epochs=35, verbose=False)


if __name__ == "__main__":
    build_and_train_cnn(MODEL_SAVE_PATH, epochs=35, verbose=True)
