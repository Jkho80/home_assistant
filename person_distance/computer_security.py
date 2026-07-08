#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
持续安全检查 - RDK-S100P BPU版本 - 人体跟踪 + command控制
使用 PTZ 服务器（ptz_server.py）进行串行化云台控制，消除竞态和累积误差。

架构：
  YOLO 检测线程（主循环）←→ 跟踪队列 ←→ 跟踪工作线程 ←→ PTZ 服务器 ←→ 云台硬件
                                   ↑ 串行化，无竞态
  主循环：YOLO 推理 + 人员检测 + 飞书告警 + 推理控制（不阻塞）
  跟踪线程：管理云台跟随运动，通过 PTZ 客户端与服务器通信

通过 command.json 可随时下发回正指令，Ctrl+C 安全退出并回正。
"""

import argparse
import subprocess
import time
import json
import datetime
import cv2
import numpy as np
import sys
import os
import logging
import signal
import socket
import threading
import queue
from pathlib import Path
from urllib import request, error, parse

from hbm_runtime import HB_HBMRuntime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger("security_check")

# 降低hbm_runtime的日志级别
logging.getLogger("hbm_runtime").setLevel(logging.WARNING)

# ---------- 数据类型映射 ----------
HB_DTYPE_MAP = {
    "U8": np.uint8, "S8": np.int8,
    "F32": np.float32, "F16": np.float16,
    "U16": np.uint16, "S16": np.int16,
    "S32": np.int32, "U32": np.uint32,
    "S64": np.int64, "U64": np.uint64,
    "BOOL8": np.bool_,
}


t_start = time.time()
    
# ============================================================
# 飞书告警 (参考 oc_error_watchdog.py)
# ============================================================

def load_env(path):
    p = Path(path).expanduser()
    if not p.exists():
        return
    for raw in p.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

FEISHU_ENV_LOADED = False
FEISHU_TOKEN_CACHE = {"token": None, "expires": 0}

def feishu_token():
    global FEISHU_TOKEN_CACHE
    app = os.environ.get('FEISHU_APP_ID', '')
    sec = os.environ.get('FEISHU_APP_SECRET', '')
    if not app or not sec:
        return None
    if FEISHU_TOKEN_CACHE.get('token') and FEISHU_TOKEN_CACHE.get('expires', 0) > time.time() + 120:
        return FEISHU_TOKEN_CACHE['token']
    base = os.environ.get('FEISHU_BASE_URL', 'https://open.feishu.cn').rstrip('/')
    try:
        req = request.Request(
            base + '/open-apis/auth/v3/tenant_access_token/internal',
            data=json.dumps({'app_id': app, 'app_secret': sec}).encode(),
            headers={'Content-Type': 'application/json; charset=utf-8'}
        )
        with request.urlopen(req, timeout=10) as resp:
            res = json.loads(resp.read().decode())
        tok = res.get('tenant_access_token')
        if tok:
            FEISHU_TOKEN_CACHE = {
                'token': tok,
                'expires': time.time() + int(res.get('expire', 7200)) - 60
            }
            return tok
    except Exception:
        pass
    return None


def send_feishu(msg_type, content_obj):
    chat = os.environ.get('FEISHU_ALARM_CHAT_ID', '')
    if not chat:
        return {'ok': False, 'reason': 'FEISHU_ALARM_CHAT_ID empty'}
    tok = feishu_token()
    if not tok:
        return {'ok': False, 'reason': 'no tenant_access_token'}
    base = os.environ.get('FEISHU_BASE_URL', 'https://open.feishu.cn').rstrip('/')
    typ = parse.quote(os.environ.get('FEISHU_RECEIVE_ID_TYPE', 'chat_id'))
    payload = {
        'receive_id': chat,
        'msg_type': msg_type,
        'content': json.dumps(content_obj, ensure_ascii=False)
    }
    try:
        req = request.Request(
            base + f'/open-apis/im/v1/messages?receive_id_type={typ}',
            data=json.dumps(payload).encode(),
            headers={
                'Authorization': 'Bearer ' + tok,
                'Content-Type': 'application/json; charset=utf-8'
            }
        )
        with request.urlopen(req, timeout=10) as resp:
            res = json.loads(resp.read().decode())
        return {'ok': res.get('code') == 0, 'response': res}
    except Exception as e:
        return {'ok': False, 'reason': str(e)}


def send_text(text):
    return send_feishu('text', {'text': text})


def send_security_alert(person_info):
    """发送安全告警到飞书"""
    content = f"**状态**：人员闯入\n**时间**：{datetime.datetime.now().isoformat(timespec='seconds')}\n**主机**：{socket.gethostname()}\n"
    if person_info:
        for i, d in enumerate(person_info[:5], 1):
            content += f"**目标{i}**：距离 {d.get('distance', '?')}m，置信度 {d.get('score', '?')}\n"
    card = {
        'config': {'wide_screen_mode': True},
        'header': {
            'template': 'red',
            'title': {'tag': 'plain_text', 'content': '⚠️ 安全警报：人员闯入操作区域'}
        },
        'elements': [
            {'tag': 'div', 'text': {'tag': 'lark_md', 'content': content}},
            {'tag': 'hr'},
            {'tag': 'div', 'text': {'tag': 'lark_md', 'content': '**自动处理**：推理已暂停，确保人员安全。请确认现场情况后恢复。'}}
        ]
    }
    result = send_feishu('interactive', card)
    if not result.get('ok'):
        send_text(f'⚠️ 安全警报：检测到人员闯入，距离 {person_info[0]["distance"]}m' if person_info else '⚠️ 安全警报：检测到人员闯入')
    return result


# ---------- 推理状态控制（线程安全）----------
_inference_lock = threading.Lock()
_inference_disabled = False   # 记录当前推理是否已被禁用
_inference_queue = queue.Queue(maxsize=10)  # 顺序队列，确保 enable/disable 按序执行

_inference_queue = queue.Queue(maxsize=10)


def _inference_worker():
    """推理状态变更工作线程：串行处理队列中的请求，保证一致性"""
    global _inference_disabled
    while True:
        try:
            enable = _inference_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        if enable is None:
            break

        # 在真正执行时检查状态（不是入队时），避免竞态
        with _inference_lock:
            if enable and not _inference_disabled:
                continue  # 当前状态已是对应状态，跳过
            if not enable and _inference_disabled:
                continue

        service = "/robot/inference_service/enable" if enable else "/robot/inference_service/disable"
        tag = "恢复" if enable else "禁用"
        try:
            cmd = [
                "bash", "-c",
                f"source /opt/ros/humble/setup.bash && "
                f"ros2 service call {service} std_srvs/srv/Trigger \"{{}}\" 2>/dev/null"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and 'success=True' in result.stdout:
                LOGGER.warning(f"\U0001f512 推理已{tag}")
                t_end = time.time()
                LOGGER.info(f"Total Time: {t_end - t_start}")
                with _inference_lock:
                    _inference_disabled = not enable
            else:
                LOGGER.warning(f"推理{tag}调用异常: {result.returncode}")
        except Exception as e:
            LOGGER.warning(f"推理{tag}异常: {e}")


# 启动推理工作线程
_inference_worker_thread = threading.Thread(
    target=_inference_worker, daemon=True, name="inference-worker"
)
_inference_worker_thread.start()


def set_inference_state(enable: bool):
    """
    请求推理状态变更。
    只负责入队，不检查当前状态（检查在 worker 执行时做）。
    不阻塞主循环。
    """
    try:
        _inference_queue.put_nowait(enable)
    except queue.Full:
        LOGGER.warning(f"推理状态队列已满，丢弃请求: {'启用' if enable else '禁用'}")



# ---------- YOLO 预处理/后处理 ----------
IMG_SIZE = 640
IOU_THRES = 0.45
PERSON_CLASS_ID = 0

def letterbox(image, new_shape=640):
    h, w = image.shape[:2]
    scale = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)
    top = (new_shape - nh) // 2
    left = (new_shape - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, scale, left, top

def preprocess_bgr(image_bgr):
    img, scale, pad_x, pad_y = letterbox(image_bgr, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img.astype(np.float32), scale, pad_x, pad_y

def nms(boxes, scores, iou_thres):
    if len(boxes) == 0:
        return []
    boxes = boxes.astype(np.float32)
    scores = scores.astype(np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter_w = np.maximum(0, xx2 - xx1)
        inter_h = np.maximum(0, yy2 - yy1)
        inter = inter_w * inter_h
        union = areas[i] + areas[order[1:]] - inter + 1e-6
        iou = inter / union
        order = order[1:][iou <= iou_thres]
    return keep

def postprocess_yolo11(outputs, scale, pad_x, pad_y, orig_w, orig_h, conf_thres):
    if isinstance(outputs, dict):
        output_name = list(outputs.keys())[0]
        predictions = outputs[output_name]
    elif isinstance(outputs, (list, tuple)):
        predictions = outputs[0]
    else:
        predictions = outputs
    predictions = np.asarray(predictions, dtype=np.float32)
    if predictions.ndim == 3:
        predictions = predictions[0]  # (1,84,8400) -> (84,8400)
    if predictions.shape[0] == 84 and predictions.shape[1] == 8400:
        predictions = predictions.T  # (8400,84)

    # 区分 84 通道（无 objectness）和 85 通道（含 objectness）
    if predictions.shape[1] == 84:
        # 4 coords + 80 classes
        boxes_xywh = predictions[:, :4]
        class_scores = predictions[:, 4:]          # (N,80)
    elif predictions.shape[1] == 85:
        # 4 coords + 1 objectness + 80 classes
        boxes_xywh = predictions[:, :4]
        obj_conf = predictions[:, 4:5]             # (N,1)
        cls_conf = predictions[:, 5:]              # (N,80)
        class_scores = obj_conf * cls_conf         # 融合置信度
    else:
        LOGGER.error(f"Unexpected predictions shape: {predictions.shape}")
        return []

    person_scores = class_scores[:, PERSON_CLASS_ID]
    mask = person_scores >= conf_thres
    if not np.any(mask):
        return []

    boxes_xywh = boxes_xywh[mask]
    person_scores = person_scores[mask]
    cx = boxes_xywh[:, 0]
    cy = boxes_xywh[:, 1]
    bw = boxes_xywh[:, 2]
    bh = boxes_xywh[:, 3]
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2
    x1 = (x1 - pad_x) / scale
    y1 = (y1 - pad_y) / scale
    x2 = (x2 - pad_x) / scale
    y2 = (y2 - pad_y) / scale
    x1 = np.clip(x1, 0, orig_w - 1)
    y1 = np.clip(y1, 0, orig_h - 1)
    x2 = np.clip(x2, 0, orig_w - 1)
    y2 = np.clip(y2, 0, orig_h - 1)
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    keep = nms(boxes, person_scores, IOU_THRES)
    dets = []
    for i in keep:
        dets.append({
            "box": boxes[i].astype(int),
            "score": float(person_scores[i])
        })
    return dets

# ---------- 人体跟踪器 ----------
class PersonTracker:
    def __init__(self, camera_fov_h=69.0, camera_width=640,
                 pan_range=(60, 120), track_speed=30.0, dead_zone=30):
        self.camera_fov_h = camera_fov_h
        self.camera_width = camera_width
        self.pan_min, self.pan_max = pan_range
        self.track_speed = track_speed
        self.dead_zone = dead_zone
        self.current_pan = None
        self.current_tilt = None

    def update_current_angle(self, pan, tilt):
        self.current_pan = pan
        self.current_tilt = tilt

    def calculate_tracking_command(self, roi_box, frame_width):
        if self.current_pan is None:
            return None
        x1, y1, x2, y2 = roi_box
        roi_center_x = (x1 + x2) / 2.0
        frame_center_x = frame_width / 2.0
        pixel_error = roi_center_x - frame_center_x
        if abs(pixel_error) <= self.dead_zone:
            return {"action": "stop", "new_pan": self.current_pan,
                    "error_pixel": pixel_error, "error_angle": 0.0}
        angle_per_pixel = self.camera_fov_h / self.camera_width
        angle_error = pixel_error * angle_per_pixel
        new_pan = self.current_pan + angle_error
        new_pan = np.clip(new_pan, self.pan_min, self.pan_max)
        return {"action": "track", "new_pan": new_pan,
                "error_pixel": pixel_error, "error_angle": angle_error}


# ---------- PTZ 跟踪工作线程 ----------
class PtzTrackingThread(threading.Thread):
    """
    跟踪工作线程：
    - 通过独立线程与 PTZ 服务器通信
    - 主检测循环将跟踪目标写入 self.track_queue
    - 此线程串行执行所有云台运动命令
    - 运动完成后更新 self.current_pan/tilt
    """

    def __init__(self, initial_pan=90.0, initial_tilt=45.0,
                 pan_min=0.0, pan_max=180.0,
                 tilt_min=0.0, tilt_max=90.0):
        super().__init__(daemon=True, name="ptz-tracking")
        self.track_queue = queue.Queue(maxsize=10)
        self.current_pan = initial_pan
        self.current_tilt = initial_tilt
        self.home_pan = initial_pan    # 保存回零位置（与 workspace 区分）
        self.home_tilt = initial_tilt
        self.pan_min = pan_min
        self.pan_max = pan_max
        self.tilt_min = tilt_min
        self.tilt_max = tilt_max
        self._running = threading.Event()
        self._running.set()
        self._busy = threading.Event()  # PTZ 正在运动中
        self.client = None
        self._connected = False

    def run(self):
        """线程主循环"""
        from ptz_client import PTZClient
        self.client = PTZClient()
        self._connected = self.client.connect()

        if not self._connected:
            LOGGER.error("❌ 跟踪线程无法连接 PTZ 服务器，跟踪功能禁用")
            # 即使无法连接，线程仍运行但跳过跟踪动作
            while self._running.is_set():
                try:
                    self.track_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
            return

        LOGGER.info("✅ 跟踪线程就绪 (已连接 PTZ 服务器)")
        LOGGER.info(f"  🏠 回零位置: pan={self.home_pan:.1f}°, tilt={self.home_tilt:.1f}°")
        LOGGER.info(f"  📋 等待主线程下发工作区位置...")

        # 主循环：从队列取跟踪目标并执行
        while self._running.is_set():
            try:
                target = self.track_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if target is None:
                break  # 退出信号

            self._busy.set()  # 标记为忙
            try:
                pan_target = target.get("pan")
                tilt_target = target.get("tilt")
                cmd_type = target.get("type", "track")

                if cmd_type == "track" and pan_target is not None:
                    # 跟踪：仅水平跟随
                    pan_target = np.clip(pan_target, self.pan_min, self.pan_max)
                    LOGGER.info(f"  🎯 跟踪 → pan={pan_target:.1f}°")
                    result = self.client.move_pan(pan_target)
                    if result.get("ok"):
                        self.current_pan = result.get("pan", self.current_pan)
                        self.current_tilt = result.get("tilt", self.current_tilt)
                    else:
                        LOGGER.warning(f"  跟踪移动异常: {result.get('error')}")

                elif cmd_type == "move_to" and pan_target is not None:
                    # 移动到指定位置
                    pan_target = np.clip(pan_target, self.pan_min, self.pan_max)
                    tilt_target = np.clip(tilt_target if tilt_target is not None else self.current_tilt,
                                          self.tilt_min, self.tilt_max)
                    LOGGER.info(f"  📍 移动到 pan={pan_target:.1f}°, tilt={tilt_target:.1f}°")
                    result = self.client.move_to(pan_target, tilt_target)
                    if result.get("ok"):
                        self.current_pan = result.get("pan", self.current_pan)
                        self.current_tilt = result.get("tilt", self.current_tilt)

                elif cmd_type == "home":
                    LOGGER.info(f"  🏠 回到初始位置 (pan={self.home_pan:.1f}°, tilt={self.home_tilt:.1f}°)")
                    result = self.client.move_to(self.home_pan, self.home_tilt)
                    if result.get("ok"):
                        self.current_pan = result.get("pan", self.current_pan)
                        self.current_tilt = result.get("tilt", self.current_tilt)

                elif cmd_type == "stop":
                    LOGGER.info("  ⏹️ 停止跟踪")
                    self.client.stop()

            except Exception as e:
                LOGGER.error(f"跟踪命令执行异常: {e}")
                # 尝试重连
                try:
                    self.client.close()
                except Exception:
                    pass
                time.sleep(0.5)
                self._connected = self.client.connect()
            finally:
                self._busy.clear()  # 无论成功/异常，都标记为空闲

        # 清理
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass
        LOGGER.info("🛑 跟踪线程已退出")

    def enqueue_track(self, pan_target):
        """将跟踪目标写入队列（非阻塞，主检测循环使用）"""
        if not self._connected:
            return
        try:
            self.track_queue.put_nowait({
                "type": "track",
                "pan": float(pan_target),
                "tilt": None,
            })
        except queue.Full:
            pass  # 队列已满时丢弃最老请求

    def enqueue_move_to(self, pan, tilt=None):
        """将移动命令写入队列"""
        if not self._connected:
            return
        # 清空队列中的旧命令（新的位置命令覆盖旧的跟踪命令）
        while not self.track_queue.empty():
            try:
                self.track_queue.get_nowait()
            except queue.Empty:
                break
        self.track_queue.put({
            "type": "move_to",
            "pan": float(pan),
            "tilt": float(tilt) if tilt is not None else None,
        })

    def enqueue_home(self):
        """将回零命令写入队列（回到 initial_pan/tilt）"""
        if not self._connected:
            return
        # 清空未执行的旧命令
        while not self.track_queue.empty():
            try:
                self.track_queue.get_nowait()
            except queue.Empty:
                break
        self.track_queue.put({"type": "home", "pan": self.home_pan, "tilt": self.home_tilt})

    def enqueue_stop(self):
        """将停止命令写入队列"""
        if not self._connected:
            return
        while not self.track_queue.empty():
            try:
                self.track_queue.get_nowait()
            except queue.Empty:
                break
        self.track_queue.put({"type": "stop"})

    def get_position(self):
        """获取当前位置（直接从 PTZ 服务器查询）"""
        if self.client and self._connected:
            try:
                result = self.client.get_position()
                if result.get("ok"):
                    self.current_pan = result.get("pan", self.current_pan)
                    self.current_tilt = result.get("tilt", self.current_tilt)
            except Exception:
                pass
        return self.current_pan, self.current_tilt

    def stop(self):
        """停止线程"""
        self._running.clear()
        self.enqueue_stop()

    def set_limits(self, pan_min=None, pan_max=None, tilt_min=None, tilt_max=None):
        """更新角度限制"""
        if pan_min is not None: self.pan_min = pan_min
        if pan_max is not None: self.pan_max = pan_max
        if tilt_min is not None: self.tilt_min = tilt_min
        if tilt_max is not None: self.tilt_max = tilt_max

    def is_busy(self):
        """跟踪线程是否忙（正在执行运动或队列非空）"""
        return self._busy.is_set() or not self.track_queue.empty()


# ---------- BPU模型类 ----------
class BPUModel:
    # ...（保持不变）...
    def __init__(self, model_path, priority=5, bpu_cores=None):
        self.model_path = model_path
        self.priority = priority
        self.bpu_cores = bpu_cores if bpu_cores is not None else [0]
        self.runtime = None
        self.model_name = None
        self.input_names = []
        self.input_shapes = {}
        self.input_dtypes = {}
        self.output_names = []
        self.output_shapes = {}
        self.output_dtypes = {}

    def load(self):
        LOGGER.info(f"Loading HBM model: {self.model_path}")
        try:
            self.runtime = HB_HBMRuntime(self.model_path)
            if not self.runtime.model_names:
                raise RuntimeError(f"No model loaded from {self.model_path}")
            self.model_name = self.runtime.model_names[0]
            self.input_names = list(self.runtime.input_names[self.model_name])
            self.input_shapes = dict(self.runtime.input_shapes[self.model_name])
            self.input_dtypes = dict(self.runtime.input_dtypes[self.model_name])
            self.output_names = list(self.runtime.output_names[self.model_name])
            self.output_shapes = dict(self.runtime.output_shapes[self.model_name])
            self.output_dtypes = dict(self.runtime.output_dtypes[self.model_name])
            self.runtime.set_scheduling_params(
                priority={self.model_name: self.priority},
                bpu_cores={self.model_name: self.bpu_cores},
            )
            LOGGER.info(f"✅ Model loaded: {self.model_name}")
            return True
        except Exception as e:
            LOGGER.error(f"Failed to load model: {e}")
            return False

    def _cast_input(self, name, arr):
        if name not in self.input_dtypes:
            return arr
        hb_dtype = self.input_dtypes[name].name
        np_dtype = HB_DTYPE_MAP.get(hb_dtype, np.float32)
        arr = np.asarray(arr)
        if arr.dtype != np_dtype:
            arr = arr.astype(np_dtype, copy=False)
        return np.ascontiguousarray(arr)

    def infer(self, feeds):
        if self.runtime is None:
            raise RuntimeError("Model not loaded")
        model_inputs = {}
        for name in self.input_names:
            if name not in feeds:
                raise KeyError(f"Missing HBM input `{name}`")
            model_inputs[name] = self._cast_input(name, feeds[name])
        t0 = time.time()
        outputs = self.runtime.run({self.model_name: model_inputs})
        t1 = time.time()
        print(f"YoloV11 Inference Time: {t1 - t0}")
        model_outputs = outputs[self.model_name]
        return {name: np.asarray(model_outputs[name]) for name in self.output_names}


# ---------- RealSense 相机类 ----------
class RealsenseCamera:
    def __init__(self, width=640, height=480, fps=30, serial=None):
        self.width = width
        self.height = height
        self.fps = fps
        self.serial = serial
        self.pipeline = None

    def start(self):
        try:
            import pyrealsense2 as rs
            ctx = rs.context()
            devices = ctx.query_devices()
            LOGGER.info(f"Found {len(devices)} RealSense device(s):")
            for i, dev in enumerate(devices):
                LOGGER.info(f"  [{i}] Serial: {dev.get_info(rs.camera_info.serial_number)}, "
                            f"Name: {dev.get_info(rs.camera_info.name)}")

            if self.serial:
                target = str(self.serial)
                found = False
                for dev in devices:
                    if dev.get_info(rs.camera_info.serial_number) == target:
                        found = True
                        break
                if not found:
                    LOGGER.error(f"Camera with serial {target} not found")
                    return False
                config = rs.config()
                config.enable_device(target)
            else:
                config = rs.config()

            self.pipeline = rs.pipeline(ctx)
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.pipeline.start(config)
                    break
                except RuntimeError as e:
                    if attempt < max_retries - 1:
                        LOGGER.warning(f"Pipeline start failed (attempt {attempt+1}/{max_retries}): {e}")
                        time.sleep(1)
                    else:
                        raise

            profile = self.pipeline.get_active_profile()
            color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
            self.intrinsics = color_profile.get_intrinsics()
            LOGGER.info(f"✅ Camera ready: {self.width}x{self.height} @ {self.fps}fps")
            return True
        except Exception as e:
            LOGGER.error(f"Failed to start RealSense: {e}")
            return False

    def get_frames(self):
        if self.pipeline:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    return None, None
                color = np.asanyarray(color_frame.get_data())
                depth = np.asanyarray(depth_frame.get_data())
                return color, depth
            except Exception as e:
                LOGGER.warning(f"Failed to get frames: {e}")
                return None, None
        return None, None

    def get_depth_meters(self, depth_frame):
        return depth_frame.astype(np.float32) * 0.001

    def stop(self):
        if self.pipeline:
            self.pipeline.stop()

def compute_distance(det, depth_m):
    x1, y1, x2, y2 = det["box"]
    h, w = depth_m.shape
    bw = x2 - x1
    bh = y2 - y1
    if bw > 0 and bh > 0:
        rx1 = max(0, int(x1 + 0.35 * bw))
        rx2 = min(w, int(x2 - 0.35 * bw))
        ry1 = max(0, int(y1 + 0.30 * bh))
        ry2 = min(h, int(y2 - 0.25 * bh))
        roi = depth_m[ry1:ry2, rx1:rx2]
        roi = roi[np.isfinite(roi)]
        roi = roi[(roi > 0.2) & (roi < 6.0)]
        if roi.size >= 30:
            return float(np.median(roi))
    return None

def write_result(timestamp, safe, detections=None):
    result = {"timestamp": timestamp, "safe": safe}
    if detections is not None and len(detections) > 0:
        result["detections"] = detections
    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)

def write_command(data):
    with open("command.json", "w") as f:
        json.dump(data, f, indent=2)

def read_command():
    try:
        with open("command.json", "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"return": False}


# ---------- 全局退出标志 ----------
exit_flag = False

def signal_handler(sig, frame):
    global exit_flag
    LOGGER.warning(f"Received signal {sig}, setting exit flag...")
    exit_flag = True


# ---------- 主程序 ----------
def main():
    global exit_flag
    global t_start

    parser = argparse.ArgumentParser(description="持续安全检查 - RDK-S100P BPU版本 - 跟踪+command控制 (使用PTZ服务器)")
    parser.add_argument("--model", default="yolo11n_person_distance_quantized.hbm")
    parser.add_argument("--port", default="/dev/ptz")
    parser.add_argument("--baudrate", type=int, default=9600)
    parser.add_argument("--conf", type=float, default=0.8)
    parser.add_argument("--initial_pan", type=float, default=90.0)
    parser.add_argument("--initial_tilt", type=float, default=45.0)
    parser.add_argument("--workspace_pan", type=float, default=30.0)  # 启动时指向工作区中心
    parser.add_argument("--workspace_tilt", type=float, default=45.0)
    parser.add_argument("--distance_threshold", type=float, default=1.5)
    parser.add_argument("--detect_interval", type=float, default=0.5, help="检测/控制间隔")
    parser.add_argument("--priority", type=int, default=5)
    parser.add_argument("--bpu_cores", type=int, nargs='+', default=[0])
    parser.add_argument("--camera_serial", type=str, default="405622074908", help="RealSense相机序列号")
    # 跟踪参数
    parser.add_argument("--pan_min", type=float, default=0.0, help="云台最小水平角度")
    parser.add_argument("--pan_max", type=float, default=90.0, help="云台最大水平角度")
    parser.add_argument("--track_speed", type=float, default=5.0, help="跟踪速度（度/秒）")
    parser.add_argument("--dead_zone", type=int, default=30, help="死区像素")
    parser.add_argument("--camera_fov_h", type=float, default=69.0, help="D435水平视场角")
    parser.add_argument("--camera_width", type=int, default=640, help="图像宽度")
    parser.add_argument("--direct-ptz", action="store_true", help="不使用 PTZ 服务器，直接控制云台（兼容旧版）")
    args = parser.parse_args()

    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 加载飞书告警环境变量
    load_env('/home/sunrise/.oc_arm/feishu_alarm.env')
    if os.environ.get('FEISHU_APP_ID'):
        LOGGER.info("📢 飞书告警已配置")
    else:
        LOGGER.warning("⚠️ 飞书告警未配置 (FEISHU_APP_ID 为空)")

    LOGGER.info("=" * 60)
    LOGGER.info("  持续安全检查系统 - RDK-S100P (PTZ 服务器模式)")
    LOGGER.info("=" * 60)

    # 检查是否有挂起的回正指令（安全启动）
    cmd = read_command()
    if cmd.get("return", False):
        LOGGER.info("📩 启动时检测到未处理的回正指令")
        write_command({"return": False})
        write_result(datetime.datetime.now().isoformat(timespec='milliseconds'), True)
        # 通过 PTZ 客户端执行回正
        try:
            from ptz_client import PTZClient
            client = PTZClient()
            if client.connect():
                client.move_to(args.initial_pan, args.initial_tilt)
                client.close()
        except Exception:
            pass
        return

    # 正常初始化状态文件
    timestamp = datetime.datetime.now().isoformat(timespec='milliseconds')
    write_result(timestamp, True)
    write_command({"return": False})
    LOGGER.info("Initialized result.json (safe) and command.json (return=false)")

    # ==================== 初始化 PTZ 服务器连接 ====================
    has_pantilt = False
    tracking_thread = None

    if args.direct_ptz:
        # 旧版：直接控制云台（兼容）
        try:
            from yuntai import PanTiltController
            pt = PanTiltController(
                port=args.port, baudrate=args.baudrate,
                initial_pan=args.initial_pan, initial_tilt=args.initial_tilt
            )
            LOGGER.info("✅ PanTiltController 直连模式（旧版）")
            has_pantilt = True

            # 移动到工作区
            current_pan = args.workspace_pan
            current_tilt = args.workspace_tilt
            LOGGER.info(f"🔍 移动到工作区位置 (pan={current_pan}°, tilt={current_tilt}°)")
            pt.set_pan(current_pan)
            pt.set_tilt(current_tilt)
            LOGGER.info(f"  ✅ 云台已到位 (pan={pt.pan:.1f}°, tilt={pt.tilt:.1f}°)")
            current_pan = pt.pan
            current_tilt = pt.tilt

            # 初始化跟踪器（使用旧版内联跟踪）
            tracker = PersonTracker(
                camera_fov_h=args.camera_fov_h,
                camera_width=args.camera_width,
                pan_range=(args.pan_min, args.pan_max),
                track_speed=args.track_speed,
                dead_zone=args.dead_zone
            )
            tracker.update_current_angle(current_pan, current_tilt)
        except Exception as e:
            LOGGER.warning(f"⚠️ 云台直连初始化失败: {e}")

    else:
        # 新版：使用 PTZ 服务器 + 跟踪线程
        try:
            from ptz_client import PTZClient

            # 先检查 PTZ 服务器是否运行
            test_client = PTZClient()
            server_running = test_client.connect()
            test_client.close()

            if not server_running:
                LOGGER.warning("⚠️ PTZ 服务器未运行，尝试自动启动...")
                import subprocess as sp
                result = sp.run(
                    [sys.executable, os.path.join(os.path.dirname(__file__), "ptz_server.py"),
                     "start",
                     "--port", args.port,
                     "--baudrate", str(args.baudrate),
                     "--initial-pan", str(args.initial_pan),
                     "--initial-tilt", str(args.initial_tilt)],
                    capture_output=True, text=True, timeout=10
                )
                LOGGER.info(f"  PTZ 服务器启动: {result.stdout.strip()}")
                time.sleep(1)

            # 创建跟踪线程
            tracking_thread = PtzTrackingThread(
                initial_pan=args.initial_pan,
                initial_tilt=args.initial_tilt,
                pan_min=args.pan_min,
                pan_max=args.pan_max,
                tilt_min=0.0,
                tilt_max=90.0,
            )
            tracking_thread.start()
            time.sleep(0.5)  # 等待跟踪线程连接到 PTZ 服务器

            # 移动到工作区位置（非阻塞，PTZ 移动和模型加载并行）
            LOGGER.info(f"🔍 移动到工作区位置 (pan={args.workspace_pan}°, tilt={args.workspace_tilt}°)")
            tracking_thread.enqueue_move_to(args.workspace_pan, args.workspace_tilt)

            has_pantilt = True
            LOGGER.info("✅ PTZ 服务器 + 跟踪线程就绪")
            LOGGER.info(f"   PTZ 正在移动到工作区 (需 ~{int(abs(args.workspace_pan - args.initial_pan)/5)}s)，与模型加载并行...")

        except Exception as e:
            LOGGER.warning(f"⚠️ PTZ 服务器初始化失败: {e}")

    # ==================== 主循环开始 ====================
    should_return_home = has_pantilt
    camera = None
    bpu_model = None
    person_ever_detected = False
    last_alert_time = 0

    try:
        # 初始化相机
        camera = RealsenseCamera(width=args.camera_width, serial=args.camera_serial)
        if not camera.start():
            LOGGER.error("❌ 相机初始化失败")
            return

        # 加载模型
        bpu_model = BPUModel(args.model, priority=args.priority, bpu_cores=args.bpu_cores)
        if not bpu_model.load():
            LOGGER.error("❌ 模型加载失败")
            return

        model_input_name = bpu_model.input_names[0] if bpu_model.input_names else "images"
        LOGGER.info(f"Model input name: {model_input_name}")

        LOGGER.info(f"🔐 开始持续安全检查 + 跟踪（跟踪范围 {args.pan_min}°-{args.pan_max}°）")
        LOGGER.info(f"   检测间隔: {args.detect_interval}s, 距离阈值: {args.distance_threshold}m")

        # 初始化跟踪器
        tracker = PersonTracker(
            camera_fov_h=args.camera_fov_h,
            camera_width=args.camera_width,
            pan_range=(args.pan_min, args.pan_max),
            track_speed=args.track_speed,
            dead_zone=args.dead_zone
        )

        # 从 PTZ 服务器获取当前位置（此时 PTZ 可能还在移动中）
        if tracking_thread:
            pan, tilt = tracking_thread.get_position()
            tracker.update_current_angle(pan, tilt)
            # 如果 PTZ 还没到位，忽略，主循环里检测到无人时会处理
            if abs(pan - args.workspace_pan) > 2.0 or tracking_thread.is_busy():
                LOGGER.info(f"  ⏳ PTZ 还在移动中 (当前 pan={pan:.1f}°)，检测循环同步启动")

        while not exit_flag:
            # 检查 command.json 主动回正指令
            cmd = read_command()
            if cmd.get("return", False):
                LOGGER.info("📩 收到返回指令，准备回正")
                write_command({"return": False})
                break

            t_start = time.time()
            color, depth = camera.get_frames()
            if color is None:
                time.sleep(0.1)
                continue

            depth_m = camera.get_depth_meters(depth)
            input_tensor, scale, pad_x, pad_y = preprocess_bgr(color)

            # 推理前再检查 command.json
            cmd = read_command()
            if cmd.get("return", False):
                LOGGER.info("📩 收到返回指令，准备回正（推理前）")
                write_command({"return": False})
                break

            # 更新跟踪器当前位置（从 PTZ 服务器查询）
            if tracking_thread:
                pan, tilt = tracking_thread.get_position()
                tracker.update_current_angle(pan, tilt)

            try:
                outputs = bpu_model.infer({model_input_name: input_tensor})
            except Exception as e:
                LOGGER.error(f"⚠️ Inference error: {e}")
                time.sleep(0.1)
                continue

            dets = postprocess_yolo11(outputs, scale, pad_x, pad_y,
                                     color.shape[1], color.shape[0], args.conf)

            close_detections = []
            best_detection = None
            for det in dets:
                dist = compute_distance(det, depth_m)
                if dist is not None and dist < args.distance_threshold:
                    close_detections.append({
                        "box": det["box"].tolist(),
                        "score": round(det["score"], 3),
                        "distance": round(dist, 2)
                    })
                    if best_detection is None or dist < best_detection.get("distance", float('inf')):
                        best_detection = {"box": det["box"], "score": det["score"], "distance": dist}

            timestamp = datetime.datetime.now().isoformat(timespec='milliseconds')
            safe = len(close_detections) == 0

            if safe:
                write_result(timestamp, True)
                # 人员离开后，重置告警标志并恢复推理
                # 要求人员至少离开 2 秒才触发（避免单帧假阴性导致状态反转）
                if last_alert_time > 0 and time.time() - last_person_time >= 2.0:
                    last_alert_time = 0
                    set_inference_state(True)
                    LOGGER.info("  ✅ 人员已离开，重置告警状态，推理已恢复")
                if not person_ever_detected:
                    LOGGER.info(f"[{timestamp}] ✅ 安全")
            else:
                # 有人
                write_result(timestamp, False, close_detections)
                person_ever_detected = True
                last_person_time = time.time()
                auto_return_triggered = False

                for d in close_detections:
                    LOGGER.warning(f"⚠️ 发现近距离目标：距离 {d['distance']:.2f}m，框={d['box']}")

                # 先跟踪（时间敏感，立即下发云台指令）
                if best_detection and tracking_thread and not tracking_thread.is_busy():
                    track_cmd = tracker.calculate_tracking_command(best_detection["box"], color.shape[1])
                    if track_cmd and track_cmd["action"] == "track":
                        new_pan = track_cmd["new_pan"]
                        LOGGER.info(f"  🎯 跟踪: 偏差 {track_cmd['error_pixel']:.1f}px → 新角度 {new_pan:.1f}°")
                        tracking_thread.enqueue_track(new_pan)

                # 后告警 + 停推理（可异步，不阻塞）
                now = time.time()
                if now - last_alert_time >= 5.0:
                    last_alert_time = now
                    LOGGER.warning("🚨 人员闯入，发送飞书告警...")
                    # TTS 语音播报
                    try:
                        with open("/tmp/oc_voice_tts.fifo", "w") as fifo:
                            fifo.write("distance_too_close\n")
                    except Exception:
                        pass
                    os.makedirs("person_images", exist_ok=True)
                    
                    # 在图像上绘制检测框
                    img_with_boxes = color.copy()
                    for det in close_detections:
                        box = det.get("box")
                        if box:
                            x1, y1, x2, y2 = box
                            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            label = f"{det.get('distance', '?')}m {det.get('score', 0):.2f}"
                            cv2.putText(img_with_boxes, label, (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    # 生成文件名
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    if best_detection:
                        dist = best_detection.get("distance", 0)
                        filename = f"person_{timestamp}_dist{dist:.2f}m.jpg"
                    else:
                        filename = f"person_{timestamp}.jpg"
                    
                    filepath = os.path.join("person_images", filename)
                    cv2.imwrite(filepath, img_with_boxes)
                    send_security_alert(close_detections)
                    set_inference_state(False)

            # 控制循环频率
            time.sleep(args.detect_interval)

    except KeyboardInterrupt:
        LOGGER.warning("\n⚠️ 检测被用户中断")
        exit_flag = True
    finally:
        # 先回零→等到位→再停 PTZ 服务器
        if should_return_home and tracking_thread:
            tracking_thread.enqueue_home()
            LOGGER.info(f"↩️ 回零中 (pan={args.initial_pan}°, tilt={args.initial_tilt}°)")
            wait_start = time.time()
            while time.time() - wait_start < 20.0:
                pan, _ = tracking_thread.get_position()
                if abs(pan - args.initial_pan) < 2.0 and not tracking_thread.is_busy():
                    LOGGER.info(f"  ✅ 回零到位: pan={pan:.1f}°")
                    break
                time.sleep(0.5)
            else:
                LOGGER.warning("  ⚠️ 回零超时")

        # 停止 PTZ 服务器进程
        LOGGER.info("⏹️ 停止 PTZ 服务器...")
        try:
            import subprocess as sp
            sp.run(
                [sys.executable, os.path.join(os.path.dirname(__file__), "ptz_server.py"), "stop"],
                capture_output=True, timeout=10
            )
        except Exception as e:
            LOGGER.warning(f"停止 PTZ 服务器失败: {e}")

        final_timestamp = datetime.datetime.now().isoformat(timespec='milliseconds')
        write_result(final_timestamp, True)
        LOGGER.info("📄 最终结果已写入 result.json")

        # 停止跟踪线程
        if tracking_thread:
            tracking_thread.stop()
            tracking_thread.join(timeout=3.0)

        if camera is not None:
            camera.stop()

        if args.direct_ptz and pt is not None:
            pt.close()

        LOGGER.info("✅ 程序正常退出")


if __name__ == "__main__":
    main()
