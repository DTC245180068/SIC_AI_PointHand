# -*- coding: utf-8 -*-
import os, sys, io, math, time, threading, ctypes, urllib.request
from collections import deque
import cv2, numpy as np, pyautogui, torch
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from train_cnn import load_or_train_cnn, extract_cnn_features, GESTURE_CLASSES

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
user32 = ctypes.windll.user32
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.0

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        user32.SetProcessDPIAware()
    except Exception:
        pass

try:
    winmm = ctypes.windll.winmm
    winmm.timeBeginPeriod(1)
except Exception:
    pass

# ── Mã phím ảo Win32 Virtual Keys (Xử lý tức thời <0.01ms, không nghẽn luồng) ───
VK_CONTROL = 0x11
VK_SHIFT   = 0x10
VK_P       = 0x50
VK_I       = 0x49
VK_E       = 0x45
VK_L       = 0x4C
VK_A       = 0x41
VK_H       = 0x48
VK_F5      = 0x74
VK_LEFT    = 0x25
VK_RIGHT   = 0x27
KEYEVENTF_KEYUP = 0x0002

MODE_RESTORE_KEYS = {
    "SWIPE": VK_A,
    "PEN": VK_P,
    "HIGHLIGHTER": VK_I,
    "ERASER": VK_E,
    "LASER": VK_L,
}

def win32_hotkey(mod_vk: int, key_vk: int):
    """Phát tổ hợp phím Win32 tức thì (<0.01ms), không trễ chuỗi PyAutoGUI."""
    user32.keybd_event(mod_vk, 0, 0, 0)
    user32.keybd_event(key_vk, 0, 0, 0)
    user32.keybd_event(key_vk, 0, KEYEVENTF_KEYUP, 0)
    user32.keybd_event(mod_vk, 0, KEYEVENTF_KEYUP, 0)

def win32_press(key_vk: int):
    """Phát phím Win32 đơn lẻ tức thì."""
    user32.keybd_event(key_vk, 0, 0, 0)
    user32.keybd_event(key_vk, 0, KEYEVENTF_KEYUP, 0)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
HAND_CONNS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
              (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
TIP_IDS = np.array([4, 8, 12, 16, 20], dtype=np.int32)
DIP_IDS = np.array([3, 7, 11, 15, 19], dtype=np.int32)
PIP_IDS = np.array([3, 6, 10, 14, 18], dtype=np.int32)
MCP_IDS = np.array([2, 5, 9, 13, 17], dtype=np.int32)

# ── Cấu hình tham số sự kiện chính (Khung cam rộng 16:9 HD) ───
CAM_W, CAM_H        = 1280, 720
WIN_W, WIN_H        = 960, 540
INFER_W, INFER_H    = 640, 360
SWIPE_DIST, SWIPE_V = 0.12, 0.12
SWIPE_COOLDOWN      = 0.85
STILL_THRESH2       = 0.032 ** 2
MARGIN_X, MARGIN_Y  = 0.06, 0.08
SMOOTH_MIN, SMOOTH_MAX, VEL_SCALE = 0.02, 0.22, 22.0

# ── Thời gian giữ (Hold Time) có chủ đích để chuyển sự kiện / công cụ ───
HOLD_PEN        = 0.90  # Giữ yên 1 ngón đủ 0.90s mới kích hoạt Bút đỏ (tránh vô tình quơ tay thành tô đỏ)
HOLD_HL         = 0.85  # Giữ yên 3 ngón đủ 0.85s mới kích hoạt Bút dạ quang
HOLD_ERASER     = 0.80  # Giữ yên 4 ngón đủ 0.80s mới kích hoạt Cục tẩy
HOLD_LASER      = 0.70  # Giữ chụm ngón (Pinch OK) đủ 0.70s mới kích hoạt Đèn Laser
HOLD_F5         = 0.90  # Giữ chữ V (2 ngón) đủ 0.90s mới bật trình chiếu Slide Show (Shift+F5)
HOLD_RETURN_PEN = 1.20  # Khi đang ở công cụ khác (Tẩy, Dạ quang, Laser), giữ 1 ngón >= 1.20s mới cho chuyển về Bút đỏ
HOLD_SWIPE_EXIT = 0.30  # Xòe 5 ngón giữ yên 0.30s để thoát về SWIPE rõ ràng


class ThreadedCamera:
    def __init__(self, idx=0):
        self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        self.opened = self.cap.isOpened()
        if self.opened:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
            self.cap.set(cv2.CAP_PROP_FPS, 60)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame, self.fid = None, 0
        self.lock = threading.Lock()
        self.new_frame = threading.Event()
        self.running = self.opened
        if self.opened:
            threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ret, f = self.cap.read()
            if ret:
                with self.lock:
                    self.frame, self.fid = f, self.fid + 1
                self.new_frame.set()
            else:
                time.sleep(0.002)

    def read(self, timeout=0.045):
        """Chờ frame phần cứng mới nhất (không lặp lại frame cũ gây giật lag)."""
        if self.new_frame.wait(timeout=timeout):
            self.new_frame.clear()
            with self.lock:
                return (self.frame is not None), self.frame
        with self.lock:
            return (self.frame is not None), self.frame

    def release(self):
        self.running = False
        self.new_frame.set()
        if self.opened:
            self.cap.release()


def fast_finger_states(pts_iso: np.ndarray, prev: np.ndarray) -> np.ndarray:
    """Vector hóa nhận diện 5 ngón tay đẳng hướng (tỷ lệ 1:1) bằng khoảng cách & góc 2 khớp (PIP + DIP)."""
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
        (d2_tip > d2_pip * np.where(prev[1:], 1.02, 1.15))
        & (d2_tip > d2_mcp * np.where(prev[1:], 1.05, 1.18))
        & (cos_pip > np.where(prev[1:], 0.15, 0.42))
        & (cos_dip > np.where(prev[1:], 0.10, 0.35))
    )

    vt1, vt2 = pts_iso[3] - pts_iso[2], pts_iso[4] - pts_iso[3]
    cos_t = float(np.dot(vt1, vt2)) / max(1e-6, float(np.linalg.norm(vt1) * np.linalg.norm(vt2)))
    d2_p_tip = float(np.sum((pts_iso[4] - pts_iso[17]) ** 2))
    d2_p_ip  = float(np.sum((pts_iso[3] - pts_iso[17]) ** 2))
    d2_m_tip = float(np.sum((pts_iso[4] - pts_iso[9]) ** 2))
    d2_m_ip  = float(np.sum((pts_iso[2] - pts_iso[9]) ** 2))
    up[0] = (
        (d2_p_tip > d2_p_ip * (1.05 if prev[0] else 1.14))
        or (d2_m_tip > d2_m_ip * (1.08 if prev[0] else 1.18))
    ) and (cos_t > (0.15 if prev[0] else 0.35))
    return up


class GestureController:
    def __init__(self, cnn_model: torch.nn.Module):
        self.cnn        = cnn_model
        self.prev_s     = np.zeros(5, dtype=bool)
        self.smooth_pts = None
        self.history    = deque(maxlen=25)
        self.last_slide = 0.0

        # Máy trạng thái cô lập (FSM): "SWIPE" | "PEN" | "HIGHLIGHTER" | "ERASER" | "LASER"
        self.mode            = "SWIPE"
        self.prev_tool       = "PEN"
        self.hl_painting     = False
        self.mouse_down      = False
        self.mode_lock_until = 0.0
        self.draw_lock_until = 0.0

        # Bộ đếm ổn định cử chỉ (chống nhiễu khi thao tác tay quá nhanh)
        self.stable_gid = -1
        self.stable_cnt = 0
        self.hl_on_cnt  = 0
        self.pause_cnt  = 0
        self.lost_cnt   = 0
        self.erase_active_cnt = 0
        self.erase_pause_cnt  = 0

        self.hold_pen = self.hold_hl = self.hold_f5 = self.hold_laser = self.hold_erase = self.hold_swipe = 0.0
        self.fist_cnt = 0
        self.cursor_hidden = False
        self.prev_px = self.prev_py = self.sx = self.sy = None
        self.cnn_pred = (-1, 0.0)
        self.last_t = time.time()
        self.sw, self.sh = pyautogui.size()

    def _set_mode(self, new_mode: str, now: float):
        """Chuyển đổi trạng thái cô lập: nhả chuột, xóa bộ đệm và phát phím Win32 tức thời."""
        self._mouse_btn(False)
        self.hl_painting = False
        self.hl_on_cnt = self.pause_cnt = self.stable_cnt = 0
        self.erase_active_cnt = self.erase_pause_cnt = 0
        self.fist_cnt = self.lost_cnt = 0
        self.hold_pen = self.hold_hl = self.hold_f5 = self.hold_laser = self.hold_erase = self.hold_swipe = 0.0
        self.history.clear()
        # Khóa nhẹ chống lặp phím khi chuyển mode (0.15s), vẽ/tẩy sẵn sàng sau 0.04s để cursor đồng bộ vị trí
        self.mode_lock_until = now + (0.08 if new_mode == "SWIPE" else 0.15)
        self.draw_lock_until = now + (0.01 if new_mode == "SWIPE" else 0.04)

        if new_mode == self.mode:
            return
        if self.mode in ("PEN", "HIGHLIGHTER"):
            self.prev_tool = self.mode
        self.mode = new_mode

        if new_mode == "PEN":
            win32_hotkey(VK_CONTROL, VK_P)
        elif new_mode == "HIGHLIGHTER":
            win32_hotkey(VK_CONTROL, VK_I)
        elif new_mode == "ERASER":
            # CHỈ GỬI Ctrl+E (Bật cục tẩy điểm PowerPoint), TUYỆT ĐỐI KHÔNG GỬI PHÍM 'E' ĐƠN LẺ
            win32_hotkey(VK_CONTROL, VK_E)
        elif new_mode == "LASER":
            win32_hotkey(VK_CONTROL, VK_L)
        elif new_mode == "SWIPE":
            win32_hotkey(VK_CONTROL, VK_A)

    def _is_pinch(self, pts_iso: np.ndarray, s: np.ndarray) -> bool:
        """Nhận diện chính xác Chụm ngón cái & ngón trỏ (Pinch 👌/🤏) theo khoảng cách đẳng hướng."""
        palm_scale = max(0.06, float(np.linalg.norm(pts_iso[9] - pts_iso[0])))
        pinch_dist = float(np.linalg.norm(pts_iso[4] - pts_iso[8]))
        d_idx_w    = float(np.linalg.norm(pts_iso[8] - pts_iso[0]))
        d_mcp_w    = float(np.linalg.norm(pts_iso[5] - pts_iso[0]))
        other_up   = int(s[2]) + int(s[3]) + int(s[4])

        # Nếu cả 4 ngón chính xòe thẳng rõ ràng thì là cử chỉ 4 hoặc 5 ngón, không phải Pinch
        if s[1] and other_up == 3 and pinch_dist > palm_scale * 0.28:
            return False

        if self.mode == "LASER":
            return (pinch_dist < palm_scale * 0.46) and (other_up >= 1 or d_idx_w > d_mcp_w * 0.98)
        return (
            (pinch_dist < palm_scale * 0.34 and other_up >= 1 and d_idx_w > d_mcp_w * 0.80)
            or (pinch_dist < palm_scale * 0.26 and other_up == 0 and d_idx_w > d_mcp_w * 1.02)
        )

    def _classify(self, pts_iso: np.ndarray, lms, label: str):
        """Phân loại lai (Hybrid): Hình học đẳng hướng 1:1 + PyTorch GestureCNN 7 lớp."""
        s = fast_finger_states(pts_iso, self.prev_s)
        pinch = self._is_pinch(pts_iso, s)

        with torch.inference_mode():
            t = torch.from_numpy(extract_cnn_features(pts_iso, label=label, aspect=1.0)).unsqueeze(0)
            probs = torch.softmax(self.cnn(t), dim=1)[0]
            cnn_cls = int(probs.argmax().item())
            cnn_conf = float(probs[cnn_cls].item())
        self.cnn_pred = (cnn_cls, cnn_conf)

        palm_scale = max(0.06, float(np.linalg.norm(pts_iso[9] - pts_iso[0])))
        pinch_dist = float(np.linalg.norm(pts_iso[4] - pts_iso[8]))
        other_up = int(s[2]) + int(s[3]) + int(s[4])
        cnn_pinch = (
            (cnn_cls == 6)
            and (cnn_conf >= 0.80)
            and (pinch_dist < palm_scale * 0.48)
            and not (s[1] and other_up == 3 and pinch_dist > palm_scale * 0.28)
        )

        if pinch or cnn_pinch:
            s[1] = False
            self.prev_s = s
            return 6, s, int(np.sum(s)), int(np.sum(s[1:])), True

        self.prev_s = s
        n_all, n_four = int(np.sum(s)), int(np.sum(s[1:]))

        if n_all == 5 or (n_four == 4 and s[0]):
            return 5, s, n_all, n_four, False
        if (cnn_cls == 5) and (cnn_conf >= 0.88) and (n_four >= 3) and (pinch_dist > palm_scale * 0.35):
            return 5, s, n_all, n_four, False

        if n_four == 4 or n_all == 4:
            return 4, s, n_all, n_four, False
        if (cnn_cls == 4) and (cnn_conf >= 0.88) and (n_four >= 3) and (pinch_dist > palm_scale * 0.35):
            return 4, s, n_all, n_four, False

        if s[1] and not s[2] and not s[3] and not s[4]:
            return 1, s, n_all, n_four, False
        if n_four in (2, 3):
            if (cnn_cls in (2, 3)) and (cnn_conf >= 0.85):
                return cnn_cls, s, n_all, n_four, False
            return n_four, s, n_all, n_four, False
        if n_four == 0:
            if (cnn_cls == 1) and (cnn_conf >= 0.90) and not s[0]:
                return 1, s, n_all, n_four, False
            return 0, s, n_all, n_four, False
        if n_four == 1:
            if (cnn_cls == 0) and (cnn_conf >= 0.90):
                return 0, s, n_all, n_four, False
            return 1, s, n_all, n_four, False
        return cnn_cls, s, n_all, n_four, False

    def _mouse_btn(self, down: bool):
        if down and not self.mouse_down:
            user32.mouse_event(0x0002, 0, 0, 0, 0)
            self.mouse_down = True
        elif not down and self.mouse_down:
            user32.mouse_event(0x0004, 0, 0, 0, 0)
            self.mouse_down = False

    def _move_cursor(self, ix: float, iy: float):
        tx = np.clip((ix - MARGIN_X) / (1.0 - 2 * MARGIN_X), 0.0, 1.0) * (self.sw - 1)
        ty = np.clip((iy - MARGIN_Y) / (1.0 - 2 * MARGIN_Y), 0.0, 1.0) * (self.sh - 1)
        if self.sx is None:
            self.sx, self.sy = float(tx), float(ty)
            px, py = int(tx), int(ty)
        else:
            px, py = int(self.sx), int(self.sy)
            vel = math.hypot(tx - self.sx, ty - self.sy) / max(1.0, self.sw)
            ret = np.clip(0.68 - vel * 14.0, 0.10, 0.68)
            self.sx = self.sx * ret + tx * (1.0 - ret)
            self.sy = self.sy * ret + ty * (1.0 - ret)

        cx, cy = int(round(self.sx)), int(round(self.sy))
        dist = math.hypot(cx - px, cy - py)
        if self.mouse_down and dist > 18:
            mx = (px + cx) // 2
            my = (py + cy) // 2
            user32.mouse_event(0x8001, int(round(mx * 65535.0 / (self.sw - 1))), int(round(my * 65535.0 / (self.sh - 1))), 0, 0)
        user32.mouse_event(0x8001, int(round(cx * 65535.0 / (self.sw - 1))), int(round(cy * 65535.0 / (self.sh - 1))), 0, 0)

    def _smooth_landmarks(self, raw_pts: np.ndarray) -> np.ndarray:
        """Lọc nhiễu thích ứng theo vận tốc cho 21 tọa độ khớp tay (giảm rung nhưng phản hồi tức thì)."""
        if self.smooth_pts is None:
            self.smooth_pts = raw_pts.copy()
        else:
            delta = np.linalg.norm(raw_pts - self.smooth_pts, axis=1, keepdims=True)
            alpha = np.clip(0.70 - delta * 15.0, 0.12, 0.70)
            self.smooth_pts = self.smooth_pts * alpha + raw_pts * (1.0 - alpha)
        return self.smooth_pts

    @staticmethod
    def _draw_hold_bar(frame: np.ndarray, w: int, h: int, progress: float, label: str = "", color=(0, 255, 180)):
        if progress > 0.02:
            bar_w = int(w * min(1.0, progress))
            cv2.rectangle(frame, (0, h - 14), (bar_w, h), color, -1)
            cv2.rectangle(frame, (0, h - 14), (w, h), (70, 70, 70), 1)
            if label:
                pct = int(min(1.0, progress) * 100)
                cv2.putText(frame, f"{label} ({pct}%)", (14, h - 22), 0, 0.55, (255, 255, 255), 3)
                cv2.putText(frame, f"{label} ({pct}%)", (14, h - 22), 0, 0.55, color, 2)

    def _hold_erase_or_laser(self, frame: np.ndarray, w: int, h: int, gid: int, still: bool, locked: bool, dt: float, now: float):
        """Chuyển sang Đèn Laser (Pinch 👌) hoặc Cục tẩy (4 ngón) khi tay giữ yên đủ thời gian."""
        if not locked and still and self.stable_cnt >= 2:
            if gid == 6:
                self.hold_laser += dt
                self.hold_erase = 0.0
                self._draw_hold_bar(frame, w, h, self.hold_laser / HOLD_LASER, "Giu chum ngon: Vao LASER", (0, 50, 255))
                if self.hold_laser >= HOLD_LASER:
                    self._set_mode("LASER", now)
            elif gid == 4:
                self.hold_erase += dt
                self.hold_laser = 0.0
                self._draw_hold_bar(frame, w, h, self.hold_erase / HOLD_ERASER, "Giu 4 ngon: Vao CUC TAY", (255, 255, 0))
                if self.hold_erase >= HOLD_ERASER:
                    self._set_mode("ERASER", now)
            else:
                self.hold_laser = max(0.0, self.hold_laser - dt * 2.5)
                self.hold_erase = max(0.0, self.hold_erase - dt * 2.5)
        else:
            self.hold_laser = max(0.0, self.hold_laser - dt * 2.5)
            self.hold_erase = max(0.0, self.hold_erase - dt * 2.5)

    def _check_swipe(self, now: float) -> int:
        if now - self.last_slide < SWIPE_COOLDOWN or now < self.mode_lock_until:
            return 0
        rec = [(x, y, t) for x, y, t in self.history if now - t <= 0.55]
        if len(rec) < 4:
            return 0
        dx, dy, dt = rec[-1][0] - rec[0][0], rec[-1][1] - rec[0][1], rec[-1][2] - rec[0][2]
        if dt <= 1e-4 or abs(dx) < SWIPE_DIST or abs(dx) / dt < SWIPE_V or abs(dx) < abs(dy) * 1.65:
            return 0
        d = 1.0 if dx > 0 else -1.0
        steps = [rec[i + 1][0] - rec[i][0] for i in range(len(rec) - 1)]
        if sum(1 for s in steps if s * d >= -0.002) / len(steps) < 0.70:
            return 0
        self.last_slide = now
        self.history.clear()
        return -1 if dx < 0 else 1

    def process(self, frame: np.ndarray, result, fps: float) -> np.ndarray:
        h, w = frame.shape[:2]
        aspect = w / max(1.0, float(h))
        now = time.time()
        dt = min(0.1, max(0.001, now - self.last_t))
        self.last_t = now

        if not result.hand_landmarks:
            if self.cursor_hidden:
                self.cursor_hidden = False
                win32_hotkey(VK_CONTROL, VK_A)
            self._mouse_btn(False)
            self.hl_painting = False
            self.smooth_pts = None
            self.prev_px = self.prev_py = None
            self.hold_pen = self.hold_hl = self.hold_f5 = self.hold_laser = self.hold_erase = self.hold_swipe = 0.0
            if self.mode != "SWIPE":
                self.lost_cnt += 1
                if self.lost_cnt >= 4:
                    self._set_mode("SWIPE", now)
            else:
                self.history.clear()
                self.prev_s[:] = False
            cv2.putText(frame, f"FPS: {int(fps)} | CON CHUOT CHUAN (Khong thay tay - Ctrl+A)", (12, 28), 0, 0.6, (0, 200, 255), 2)
            return frame

        self.lost_cnt = 0
        lms = result.hand_landmarks[0]
        label = "Right" if not result.handedness else result.handedness[0][0].category_name
        raw_pts = np.fromiter((c for lm in lms for c in (lm.x, lm.y)), dtype=np.float32, count=42).reshape(21, 2)
        pts = self._smooth_landmarks(raw_pts)
        pts_iso = pts * np.array([aspect, 1.0], dtype=np.float32)

        gid, s, n_all, n_four, pinch = self._classify(pts_iso, lms, label)

        # Đếm độ ổn định cử chỉ liên tiếp (chống xung đột khi bung/gập ngón quá nhanh)
        if gid == self.stable_gid:
            self.stable_cnt += 1
        else:
            self.stable_gid, self.stable_cnt = gid, 1
            # Khi đổi sang cử chỉ khác: Lập tức reset các timer của cử chỉ cũ về 0.0 (Zero Cross-Talk)
            if gid != 1: self.hold_pen = 0.0
            if gid != 2: self.hold_f5 = 0.0
            if gid != 3: self.hold_hl = 0.0
            if gid != 4: self.hold_erase = 0.0
            if gid != 6: self.hold_laser = 0.0

        palm = pts[[0, 5, 9, 13, 17]].mean(axis=0)
        # Cô lập tuyệt đối: Chỉ lưu lịch sử chuyển động khi đang ở SWIPE, giơ 5 ngón và không giữ timer nào
        any_hold = max(self.hold_pen, self.hold_hl, self.hold_f5, self.hold_erase, self.hold_laser) > 0.0
        if self.mode == "SWIPE" and gid == 5 and not any_hold:
            self.history.append((float(palm[0]), float(palm[1]), now))
        else:
            self.history.clear()

        # Vẽ khung xương và 21 điểm đánh dấu tọa độ chính xác trên từng khớp tay
        px = np.round(pts * np.array([w, h], dtype=np.float32)).astype(np.int32)
        for a, b in HAND_CONNS:
            cv2.line(frame, tuple(px[a]), tuple(px[b]), (0, 200, 80), 2, cv2.LINE_AA)
        tip_lookup = {4: 0, 8: 1, 12: 2, 16: 3, 20: 4}
        mcp_set = {0, 5, 9, 13, 17}
        for i, (cx, cy) in enumerate(px):
            cxy = (int(cx), int(cy))
            if i in tip_lookup:
                is_up = bool(s[tip_lookup[i]])
                fill = (0, 255, 255) if is_up else (0, 140, 255)
                cv2.circle(frame, cxy, 7 if is_up else 5, fill, -1, cv2.LINE_AA)
                cv2.circle(frame, cxy, 8 if is_up else 6, (255, 255, 255), 1, cv2.LINE_AA)
            elif i in mcp_set:
                cv2.circle(frame, cxy, 5, (255, 160, 0), -1, cv2.LINE_AA)
                cv2.circle(frame, cxy, 6, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, cxy, 4, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, cxy, 5, (255, 255, 255), 1, cv2.LINE_AA)

        # Chọn tọa độ điều khiển chuột / tâm chụm Laser
        if pinch or self.mode == "LASER":
            ix, iy = float((pts[4, 0] + pts[8, 0]) * 0.5), float((pts[4, 1] + pts[8, 1]) * 0.5)
        elif self.mode == "ERASER":
            # Cục tẩy luôn bám theo đầu ngón trỏ, TUYỆT ĐỐI không nhảy vào lòng bàn tay (pts[9])
            ix, iy = float(pts[8, 0]), float(pts[8, 1])
        else:
            if s[1] or (self.mode == "PEN" and self.mouse_down):
                tip_id = 8
            elif n_four == 1:
                tip_id = int(TIP_IDS[int(np.flatnonzero(s[1:])[0]) + 1])
            elif n_all == 1:
                tip_id = int(TIP_IDS[int(np.flatnonzero(s)[0])])
            else:
                tip_id = 8
            ix, iy = float(pts[tip_id, 0]), float(pts[tip_id, 1])

        # Vẽ tâm điểm điều khiển đang bám trên camera (Laser đỏ, Highlight vàng, Tẩy trắng)
        tx_px, ty_px = int(round(ix * w)), int(round(iy * h))
        if pinch or self.mode == "LASER":
            cv2.circle(frame, (tx_px, ty_px), 9, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (tx_px, ty_px), 13, (255, 255, 255), 2, cv2.LINE_AA)
        elif self.mode == "ERASER":
            cv2.circle(frame, (tx_px, ty_px), 12, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(frame, (tx_px, ty_px), 4, (255, 255, 255), -1, cv2.LINE_AA)
        elif self.mode != "SWIPE" and not self.cursor_hidden:
            cv2.circle(frame, (tx_px, ty_px), 9, (0, 255, 255), 2, cv2.LINE_AA)

        # Đo độ tĩnh theo tâm lòng bàn tay để gập/duỗi ngón chuyển chế độ nhạy tức thì
        px_c, py_c = float(palm[0]), float(palm[1])
        if self.prev_px is not None:
            dx_p = (px_c - self.prev_px) * aspect
            dy_p = py_c - self.prev_py
            still = (dx_p * dx_p + dy_p * dy_p) < STILL_THRESH2
        else:
            still = False
        self.prev_px, self.prev_py = px_c, py_c

        locked = now < self.mode_lock_until
        can_draw = now >= self.draw_lock_until
        msg, color = "", (0, 255, 0)

        # ── XỬ LÝ ẨN / HIỆN CON TRỎ CHUỘT KHI GIƠ 5 NGÓN TAY (Ctrl+H / Ctrl+A) ──
        if gid == 5:
            if not self.cursor_hidden:
                self.cursor_hidden = True
                self._mouse_btn(False)
                self.hl_painting = False
                win32_hotkey(VK_CONTROL, VK_H)
                user32.mouse_event(0x8001, 65535, 65535, 0, 0)
                self.sx = self.sy = None
        else:
            if self.cursor_hidden:
                self.cursor_hidden = False
                mod_key = MODE_RESTORE_KEYS.get(self.mode, VK_A)
                win32_hotkey(VK_CONTROL, mod_key)

        # ── KHI KHÔNG THẤY NGÓN TAY NÀO (0 NGÓN / NẮM TAY) ──
        # Tự động chuyển về trạng thái con chuột chuẩn (Ctrl+A), không nằm trong bất kỳ sự kiện nào cả!
        is_true_fist = (n_all == 0) or (gid == 0 and n_four == 0)
        no_fingers   = is_true_fist or (gid == 0) or (n_four == 0 and not s[0])
        if no_fingers:
            self.fist_cnt += 1
            if is_true_fist or self.fist_cnt >= 2:
                self._mouse_btn(False)
                self.hl_painting = False
                self.hold_pen = self.hold_hl = self.hold_f5 = self.hold_laser = self.hold_erase = self.hold_swipe = 0.0
            if self.mode != "SWIPE" and self.fist_cnt >= 4:
                self._set_mode("SWIPE", now)
            if self.mode == "SWIPE" or self.fist_cnt >= 2:
                msg, color = "TRANG THAI CON CHUOT (Mui ten - Ctrl+A)", (0, 200, 255)
        else:
            self.fist_cnt = 0

            # ══════════════════════════════════════════════════════════════════
            # MÁY TRẠNG THÁI CÔ LẬP 5 SỰ KIỆN CHÍNH (KHÔNG XUNG ĐỘT KHI THAO TÁC NHANH)
            # ══════════════════════════════════════════════════════════════════

            # ── ƯU TIÊN SỐ 1: XÒE 5 NGÓN TAY -> THOÁT VỀ CHẾ ĐỘ TUA SLIDE (SWIPE) ──
            if self.mode != "SWIPE" and gid == 5:
                self._mouse_btn(False)
                self.hl_painting = False
                self.hold_swipe += dt
                self._draw_hold_bar(frame, w, h, self.hold_swipe / HOLD_SWIPE_EXIT, "Xoe 5 ngon: Thoat ve TUA SLIDE", (0, 140, 255))
                if self.hold_swipe >= HOLD_SWIPE_EXIT:
                    self._set_mode("SWIPE", now)
                    msg, color = "THOAT VE CHE DO TUA SLIDE (5 ngon)", (0, 140, 255)
            else:
                self.hold_swipe = 0.0

                # ── TRẠNG THÁI 0: ĐÈN LASER RỌI SLIDE (CHỤM NGÓN CÁI & TRỎ 👌 - Ctrl+L) ──
                if self.mode == "LASER":
                    self._mouse_btn(False)
                    self._move_cursor(ix, iy)
                    msg, color = "DANG ROI DEN LASER (Pinch OK - Ctrl+L)", (0, 50, 255)

                    palm_s = max(0.06, float(np.linalg.norm(pts_iso[9] - pts_iso[0])))
                    pinch_d = float(np.linalg.norm(pts_iso[4] - pts_iso[8]))
                    clear_index = (not pinch) and (gid == 1) and (n_four == 1) and s[1] and (pinch_d > palm_s * 0.55)

                    if not locked and not pinch and still and self.stable_cnt >= 2:
                        if gid == 4:
                            self.hold_erase += dt
                            self.hold_pen = self.hold_hl = 0.0
                            self._draw_hold_bar(frame, w, h, self.hold_erase / HOLD_ERASER, "Giu 4 ngon: Vao CUC TAY", (255, 255, 0))
                            if self.hold_erase >= HOLD_ERASER:
                                self._set_mode("ERASER", now)
                        elif gid == 3 and n_four == 3:
                            self.hold_hl += dt
                            self.hold_pen = self.hold_erase = 0.0
                            self._draw_hold_bar(frame, w, h, self.hold_hl / HOLD_HL, "Giu 3 ngon: Vao DA QUANG", (0, 255, 255))
                            if self.hold_hl >= HOLD_HL:
                                self._set_mode("HIGHLIGHTER", now)
                        elif clear_index:
                            self.hold_pen += dt
                            self.hold_hl = self.hold_erase = 0.0
                            self._draw_hold_bar(frame, w, h, self.hold_pen / HOLD_RETURN_PEN, "Giu 1 ngon: Chuyen sang BUT DO", (0, 80, 255))
                            if self.hold_pen >= HOLD_RETURN_PEN:
                                self._set_mode("PEN", now)
                        else:
                            self.hold_erase = max(0.0, self.hold_erase - dt * 2.5)
                            self.hold_pen   = max(0.0, self.hold_pen   - dt * 2.5)
                            self.hold_hl    = max(0.0, self.hold_hl    - dt * 2.5)
                    else:
                        self.hold_erase = max(0.0, self.hold_erase - dt * 2.5)
                        self.hold_pen   = max(0.0, self.hold_pen   - dt * 2.5)
                        self.hold_hl    = max(0.0, self.hold_hl    - dt * 2.5)

                # ── TRẠNG THÁI 1: CỤC TẨY XÓA THEO VỊ TRÍ CHUỘT (4 NGÓN - Ctrl+E) ──
                elif self.mode == "ERASER":
                    is_erasing = can_draw and ((gid == 4) or (n_four >= 4) or (self.mouse_down and n_four >= 2 and gid != 1))
                    self._move_cursor(ix, iy)

                    if is_erasing:
                        self.erase_active_cnt += 1
                        self.erase_pause_cnt = 0
                        self._mouse_btn(True)
                        msg, color = "DANG TAY XOA TAI DIEM CHAM (4 ngon)", (255, 255, 0)
                    else:
                        self.erase_pause_cnt += 1
                        self.erase_active_cnt = 0
                        # Nhả chuột khi chủ động chuyển về 1 ngón ngắm hoặc sau 2 frame dừng xòe tay
                        if self.erase_pause_cnt >= 2 or (n_four <= 1 and s[1]):
                            self._mouse_btn(False)

                        if not locked and still and self.stable_cnt >= 2:
                            if gid == 6:
                                self.hold_laser += dt
                                self.hold_pen = self.hold_hl = 0.0
                                self._draw_hold_bar(frame, w, h, self.hold_laser / HOLD_LASER, "Giu chum ngon: Vao LASER", (0, 50, 255))
                                if self.hold_laser >= HOLD_LASER:
                                    self._set_mode("LASER", now)
                            elif n_four in (2, 3):
                                self.hold_hl += dt
                                self.hold_pen = self.hold_laser = 0.0
                                self._draw_hold_bar(frame, w, h, self.hold_hl / HOLD_HL, "Giu 3 ngon: Vao DA QUANG", (0, 255, 255))
                                if self.hold_hl >= HOLD_HL:
                                    self._set_mode("HIGHLIGHTER", now)
                            elif n_four == 1 and s[1]:
                                self.hold_pen += dt
                                self.hold_laser = self.hold_hl = 0.0
                                self._draw_hold_bar(frame, w, h, self.hold_pen / HOLD_RETURN_PEN, "Giu 1 ngon: Chuyen sang BUT DO", (0, 80, 255))
                                if self.hold_pen >= HOLD_RETURN_PEN:
                                    self._set_mode("PEN", now)
                            else:
                                self.hold_laser = self.hold_pen = self.hold_hl = 0.0
                                msg, color = "DI CHUOT CUC TAY (Chua an xoa)", (200, 220, 0)
                        else:
                            self.hold_laser = max(0.0, self.hold_laser - dt * 2.5)
                            self.hold_pen   = max(0.0, self.hold_pen   - dt * 2.5)
                            self.hold_hl    = max(0.0, self.hold_hl    - dt * 2.5)
                            msg, color = "DI CHUOT CUC TAY (Chua an xoa)", (200, 220, 0)

                # ── TRẠNG THÁI 2: BÚT DẠ QUANG (HIGHLIGHTER - Ctrl+I) ──
                elif self.mode == "HIGHLIGHTER":
                    if gid in (4, 6) or n_four >= 4:
                        self.hl_painting = False
                        self.hl_on_cnt = 0
                        self._mouse_btn(False)
                        self._hold_erase_or_laser(frame, w, h, gid, still, locked, dt, now)
                    else:
                        self.hold_laser = self.hold_erase = 0.0
                        if n_four in (2, 3) or gid in (2, 3):
                            self.hl_on_cnt += 1
                            self.pause_cnt = 0
                            self.hold_pen = 0.0
                            if can_draw:
                                self.hl_painting = True
                        else:
                            self.pause_cnt += 1
                            if self.pause_cnt >= 4 or (n_four <= 1 and s[1]):
                                self.hl_on_cnt = 0
                                self.hl_painting = False

                        self._move_cursor(ix, iy)
                        self._mouse_btn(self.hl_painting and can_draw)
                        msg   = "DANG TO DA QUANG (>=2 ngon)" if self.hl_painting else "DI CHUOT DA QUANG (1 ngon)"
                        color = (0, 255, 255) if self.hl_painting else (0, 220, 180)

                        if not locked and not self.hl_painting and still and n_four == 1 and s[1]:
                            self.hold_pen += dt
                            self._draw_hold_bar(frame, w, h, self.hold_pen / HOLD_RETURN_PEN, "Giu 1 ngon: Chuyen sang BUT DO", (0, 80, 255))
                            if self.hold_pen >= HOLD_RETURN_PEN:
                                self._set_mode("PEN", now)
                        else:
                            self.hold_pen = max(0.0, self.hold_pen - dt * 2.5)

                # ── TRẠNG THÁI 3: BÚT MỰC ĐỎ (PEN - Ctrl+P) ──
                elif self.mode == "PEN":
                    if gid in (4, 6) or n_four >= 4:
                        self._mouse_btn(False)
                        self.hold_hl = 0.0
                        self._hold_erase_or_laser(frame, w, h, gid, still, locked, dt, now)
                    else:
                        self.hold_laser = self.hold_erase = 0.0
                        if not locked and still and n_four in (2, 3) and self.stable_cnt >= 2:
                            self._mouse_btn(False)
                            self.hold_hl += dt
                            self._draw_hold_bar(frame, w, h, self.hold_hl / HOLD_HL, "Giu 3 ngon: Vao DA QUANG", (0, 255, 255))
                            if self.hold_hl >= HOLD_HL:
                                self._set_mode("HIGHLIGHTER", now)
                        else:
                            self.hold_hl = max(0.0, self.hold_hl - dt * 2.5)

                        is_drawing = can_draw and (gid == 1 or (s[1] and n_four <= 2))
                        self._move_cursor(ix, iy)
                        if is_drawing:
                            self.pause_cnt = 0
                            self._mouse_btn(True)
                        else:
                            self.pause_cnt += 1
                            if self.pause_cnt >= 4:
                                self._mouse_btn(False)

                        msg   = "DANG VE MUC DO (1 ngon)" if self.mouse_down else "TAM DUNG VE"
                        color = (0, 80, 255) if self.mouse_down else (0, 220, 120)

                # ── TRẠNG THÁI 4: ĐIỀU HƯỚNG SLIDE & KÍCH HOẠT CÔNG CỤ (SWIPE - Ctrl+A) ──
                else:
                    if not self.cursor_hidden:
                        self._move_cursor(ix, iy)
                    else:
                        if not msg:
                            msg, color = "AN CON TRO CHUOT (5 ngon - Ctrl+H)", (0, 200, 255)
                    # Cô lập hoàn toàn các bộ đếm thời gian giữ: Chỉ cho phép DUY NHẤT 1 bộ đếm chạy tại một thời điểm
                    if not locked and still and gid in (1, 2, 3, 4, 6) and self.stable_cnt >= 2:
                        if gid == 1:
                            self.hold_pen += dt
                            self.hold_f5 = self.hold_hl = self.hold_erase = self.hold_laser = 0.0
                            self._draw_hold_bar(frame, w, h, self.hold_pen / HOLD_PEN, "Giu 1 ngon: Vao BUT DO", (0, 80, 255))
                            if self.hold_pen >= HOLD_PEN:
                                self._set_mode("PEN", now)
                        elif gid == 2:
                            self.hold_f5 += dt
                            self.hold_pen = self.hold_hl = self.hold_erase = self.hold_laser = 0.0
                            self._draw_hold_bar(frame, w, h, self.hold_f5 / HOLD_F5, "Giu chu V: BAT TRINH CHIEU (Shift+F5)", (0, 255, 120))
                            if self.hold_f5 >= HOLD_F5:
                                self.hold_f5 = 0.0
                                self.mode_lock_until = now + 0.60
                                self.history.clear()
                                win32_hotkey(VK_SHIFT, VK_F5)
                                msg = "BAT TRINH CHIEU (Shift+F5)"
                        elif gid == 3:
                            self.hold_hl += dt
                            self.hold_pen = self.hold_f5 = self.hold_erase = self.hold_laser = 0.0
                            self._draw_hold_bar(frame, w, h, self.hold_hl / HOLD_HL, "Giu 3 ngon: Vao DA QUANG", (0, 255, 255))
                            if self.hold_hl >= HOLD_HL:
                                self._set_mode("HIGHLIGHTER", now)
                        elif gid == 4:
                            self.hold_erase += dt
                            self.hold_pen = self.hold_f5 = self.hold_hl = self.hold_laser = 0.0
                            self._draw_hold_bar(frame, w, h, self.hold_erase / HOLD_ERASER, "Giu 4 ngon: Vao CUC TAY", (255, 255, 0))
                            if self.hold_erase >= HOLD_ERASER:
                                self._set_mode("ERASER", now)
                        elif gid == 6:
                            self.hold_laser += dt
                            self.hold_pen = self.hold_f5 = self.hold_hl = self.hold_erase = 0.0
                            self._draw_hold_bar(frame, w, h, self.hold_laser / HOLD_LASER, "Giu chum ngon: Vao LASER", (0, 50, 255))
                            if self.hold_laser >= HOLD_LASER:
                                self._set_mode("LASER", now)
                        else:
                            self.hold_pen = max(0.0, self.hold_pen - dt * 2.0)
                            self.hold_f5  = max(0.0, self.hold_f5  - dt * 2.0)
                            self.hold_hl  = max(0.0, self.hold_hl  - dt * 2.0)
                            self.hold_erase = max(0.0, self.hold_erase - dt * 2.0)
                            self.hold_laser = max(0.0, self.hold_laser - dt * 2.0)
                    else:
                        self.hold_pen = max(0.0, self.hold_pen - dt * 2.0)
                        self.hold_f5  = max(0.0, self.hold_f5  - dt * 2.0)
                        self.hold_hl  = max(0.0, self.hold_hl  - dt * 2.0)
                        self.hold_erase = max(0.0, self.hold_erase - dt * 2.0)
                        self.hold_laser = max(0.0, self.hold_laser - dt * 2.0)

                    # Chỉ cho phép vuốt chuyển slide khi đang xòe 5 ngón và không giữ kích hoạt công cụ nào
                    if gid == 5 and max(self.hold_pen, self.hold_hl, self.hold_f5, self.hold_erase, self.hold_laser) == 0.0:
                        sw = self._check_swipe(now)
                        if sw == -1:
                            self.history.clear()
                            win32_press(VK_RIGHT)
                            msg, color = ">>> TIEN SLIDE", (0, 200, 255)
                        elif sw == 1:
                            self.history.clear()
                            win32_press(VK_LEFT)
                            msg, color = "<<< LUI SLIDE", (0, 200, 255)

        g_name = GESTURE_CLASSES.get(gid, "")
        _, cnn_p = self.cnn_pred
        if self.cursor_hidden:
            mode_label = "CON TRO AN"
        else:
            mode_label = "CON CHUOT" if self.mode == "SWIPE" else self.mode
        cv2.putText(frame, f"FPS: {int(fps)} | [{mode_label}] {g_name} (DL:{int(cnn_p * 100)}%)", (12, 26), 0, 0.52, (0, 255, 0), 2)
        if msg:
            cv2.putText(frame, msg, (12, h - 16), 0, 0.65, color, 2)
        return frame


def main():
    torch.set_num_threads(1)
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    landmarker = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.50,
        )
    )
    cam = ThreadedCamera(0)
    if not cam.opened:
        return

    ctrl = GestureController(load_or_train_cnn())
    win = "Gesture Slide Control"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, WIN_W, WIN_H)
    cv2.moveWindow(win, max(0, ctrl.sw - WIN_W - 20), 20)

    start_t = fps_t = time.time()
    last_ts, frames, fps, styled = -1, 0, 60.0, False

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(cv2.resize(frame, (INFER_W, INFER_H), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
            ts = max(last_ts + 1, int((time.time() - start_t) * 1000))
            last_ts = ts

            res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
            frames += 1
            if time.time() - fps_t >= 0.5:
                fps, frames, fps_t = frames / (time.time() - fps_t), 0, time.time()

            cv2.imshow(win, ctrl.process(frame, res, fps))

            if not styled:
                hwnd = user32.FindWindowW(None, win)
                if hwnd:
                    s = user32.GetWindowLongW(hwnd, -20)
                    user32.SetWindowLongW(hwnd, -20, s | 0x00080000 | 0x00000020 | 0x08000000)
                    user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0013)
                    styled = True

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), ord("Q")) or (user32.GetAsyncKeyState(ord("Q")) & 0x8000):
                break
    except KeyboardInterrupt:
        pass
    finally:
        ctrl._mouse_btn(False)
        cam.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()
