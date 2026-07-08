#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
表情识别 3 秒轮询 — RDK BPU HBM版 
自启动 RealSense D435I 相机，每 ~3 秒检测人脸并进行表情识别。
当判定为 sad 时触发：飞书通知 + TTS 语音 + 启动摇瓶子工作流 + 退出。
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs

# 导入地平线 HBM Runtime
try:
    from hbm_runtime import HB_HBMRuntime
except ImportError:
    print("[ERROR] 未找到 hbm_runtime 模块，请确认 RDK 环境配置正确", file=sys.stderr)
    sys.exit(1)

# ── 常量 ──────────────────────────────────────────────────────────────
EMOTIONS = ["angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
SAD_INDEX = EMOTIONS.index("sad")

# ── 推理缓存与调试保存 ──────────────────────────────────────────────
EMOTION_CACHE_DIR = os.path.expanduser("~/Face_Emotion/emotion_cache/")
DEBUG_FRAME_DIR = os.path.join(EMOTION_CACHE_DIR, "debug_frames/")
os.makedirs(EMOTION_CACHE_DIR, exist_ok=True)
os.makedirs(DEBUG_FRAME_DIR, exist_ok=True)

def save_emotion_cache(data: dict) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    fpath = os.path.join(EMOTION_CACHE_DIR, f"emotion_{ts}.json")
    try:
        with open(fpath, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return fpath
    except Exception as exc:
        print(f"[CACHE ERROR] 写入 JSON 失败: {exc}", file=sys.stderr)
        return ""

def save_sad_image(frame_bgr: np.ndarray, boxes: np.ndarray,
                   face_crop: Optional[np.ndarray] = None,
                   label: str = "sad") -> Tuple[str, str]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prefix = os.path.join(EMOTION_CACHE_DIR, f"sad_{ts}")
    full_path = prefix + ".jpg"
    crop_path = prefix + "_crop.jpg"
    try:
        vis = frame_bgr.copy()
        for bi in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[bi])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(vis, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        cv2.imwrite(full_path, vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
    except Exception:
        full_path = ""
    try:
        if face_crop is not None and face_crop.size:
            cv2.imwrite(crop_path, face_crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        else:
            crop_path = ""
    except Exception:
        crop_path = ""
    return full_path, crop_path

# ── 常量数据 ──────────────────────────────────────────────────────────
QUEUE_FILE = "/tmp/oc_voice_queue.jsonl"
CAMERA_SERIAL = "405622074908"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# ── 通用图像处理函数 ──────────────────────────────────────────────────
def letterbox_bgr(img: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)):
    sh, sw = img.shape[:2]
    dh, dw = new_shape
    scale = min(dh / sh, dw / sw)
    nu = (int(round(sw * scale)), int(round(sh * scale)))
    dx, dy = (dw - nu[0]) / 2, (dh - nu[1]) / 2
    res = cv2.resize(img, nu, interpolation=cv2.INTER_LINEAR)
    t, b = int(round(dy - 0.1)), int(round(dy + 0.1))
    l, rp = int(round(dx - 0.1)), int(round(dx + 0.1))
    return cv2.copyMakeBorder(res, t, b, l, rp, cv2.BORDER_CONSTANT, value=color), scale, (dx, dy)

def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32).reshape(-1)
    x -= np.max(x)
    e = np.exp(x)
    return e / max(float(np.sum(e)), 1e-12)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.clip(x.astype(np.float32), -60, 60)))

def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thres: float, max_det: int) -> List[int]:
    if len(boxes) == 0: return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0., x2 - x1) * np.maximum(0., y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0 and len(keep) < max_det:
        i = int(order[0])
        keep.append(i)
        if order.size == 1: break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0., xx2 - xx1) * np.maximum(0., yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-12)
        order = order[1:][iou <= iou_thres]
    return keep

def crop_face_112(frame: np.ndarray, box: np.ndarray, expand=0.18):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box.astype(float)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 2 or bh <= 2: return None
    x1 -= bw * expand
    x2 += bw * expand
    y1 -= bh * expand * 1.2
    y2 += bh * expand
    x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
    x2, y2 = min(w - 1, int(round(x2))), min(h - 1, int(round(y2)))
    if x2 <= x1 or y2 <= y1: return None
    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR) if crop.size else None

# ── HBM 模型封装类 ────────────────────────────────────────────────────
class HBMYoloFace:
    def __init__(self, model_path: str, input_size=640, conf_thres=0.45, iou_thres=0.45, max_det=1):
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det 
        self.model_name = None
        
        self.runtime = HB_HBMRuntime(model_path)
        if not self.runtime.model_names:
            raise RuntimeError(f"未从 {model_path} 加载到模型")
        self.model_name = self.runtime.model_names[0]
        
        self.input_names = list(self.runtime.input_names[self.model_name])
        self.output_names = list(self.runtime.output_names[self.model_name])
        print(f"[HBM] 模型加载成功: {self.model_name}")
        print(f"[HBM] Inputs: {self.input_names}, Outputs: {self.output_names}")

    def _postprocess(self, outputs, scale, pad_x, pad_y, orig_w, orig_h):
        # ===== 💡 修复点：正确剥掉 hbm_runtime 的多层嵌套字典 =====
        if self.model_name in outputs and self.output_names[0] in outputs[self.model_name]:
            # 标准格式：outputs['模型名']['输出层名']
            predictions = outputs[self.model_name][self.output_names[0]]
        elif isinstance(outputs, dict):
            # 容错处理：尝试剥一层字典
            first_val = list(outputs.values())[0]
            if isinstance(first_val, dict):
                predictions = list(first_val.values())[0]
            else:
                predictions = first_val
        else:
            predictions = outputs
            
        predictions = np.asarray(predictions, dtype=np.float32)
        
        if predictions.ndim == 3:
            predictions = predictions[0]
            
        if predictions.shape[0] in (84, 85) and predictions.shape[1] > 1:
            predictions = predictions.T

        if predictions.shape[1] == 84:
            boxes_xywh = predictions[:, :4]
            class_scores = predictions[:, 4:]          # (N,80)
        elif predictions.shape[1] == 85:
            boxes_xywh = predictions[:, :4]
            obj_conf = predictions[:, 4:5]             # (N,1)
            cls_conf = predictions[:, 5:]              # (N,80)
            class_scores = obj_conf * cls_conf         # 融合置信度
        else:
            return np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32)

        # COCO 类别索引 0 是 "Person"
        person_scores = class_scores[:, 0]
        mask = person_scores >= self.conf_thres
        if not np.any(mask):
            return np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32)

        boxes_xywh = boxes_xywh[mask]
        person_scores = person_scores[mask]
        
        cx, cy, bw, bh = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
        
        # 正确的逆缩放还原回原图坐标
        x1 = (cx - bw / 2 - pad_x) / scale
        y1 = (cy - bh / 2 - pad_y) / scale
        x2 = (cx + bw / 2 - pad_x) / scale
        y2 = (cy + bh / 2 - pad_y) / scale
        
        x1 = np.clip(x1, 0, orig_w - 1)
        y1 = np.clip(y1, 0, orig_h - 1)
        x2 = np.clip(x2, 0, orig_w - 1)
        y2 = np.clip(y2, 0, orig_h - 1)
        
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        
        # NMS 过滤
        keep = nms_xyxy(boxes, person_scores, self.iou_thres, self.max_det)
        return boxes[keep], person_scores[keep]

    def detect(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        img, scale, (pad_x, pad_y) = letterbox_bgr(frame_bgr, (self.input_size, self.input_size))
        blob = np.transpose(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255., (2, 0, 1))[None]
        
        # 使用高精度 perf_counter 计时
        t0 = time.perf_counter()
        feeds = {self.input_names[0]: blob}
        outputs = self.runtime.run({self.model_name: feeds})
        t1 = time.perf_counter()
        infer_ms = (t1 - t0) * 1000.0

        boxes, scores = self._postprocess(
            outputs, 
            scale=scale, 
            pad_x=pad_x, 
            pad_y=pad_y, 
            orig_w=frame_bgr.shape[1], 
            orig_h=frame_bgr.shape[0]
        )
        return boxes, scores, infer_ms

# ── 表情识别模型类 ─────────────────────────────────────────────────────
class MobileFaceNetFERONNX:
    def __init__(self, model_path: str):
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def predict(self, face_bgr_112: np.ndarray) -> Tuple[str, float, np.ndarray, float]:
        img = cv2.cvtColor(face_bgr_112, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        img = (img - 0.5) / 0.5
        blob = np.transpose(img, (2, 0, 1))[None].astype(np.float32)
        _t0 = time.perf_counter()
        logits = self.session.run(self.output_names, {self.input_name: blob})[0].reshape(-1)
        _t1 = time.perf_counter()
        _infer_ms = (_t1 - _t0) * 1000.0
        probs = softmax(logits)
        idx = int(np.argmax(probs))
        return EMOTIONS[idx], float(probs[idx]), probs, _infer_ms

# ── RealSense 相机 ─────────────────────────────────────────────────────
class RealsenseCamera:
    def __init__(self, serial: str, width=640, height=480, fps=30):
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None

    def start(self) -> bool:
        try:
            ctx = rs.context()
            config = rs.config()
            config.enable_device(self.serial)
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            self.pipeline = rs.pipeline(ctx)
            self.pipeline.start(config)
            print(f"[CAM] 相机启动: serial={self.serial}")
            return True
        except Exception as exc:
            print(f"[CAM ERROR] 相机启动失败: {exc}")
            return False

    def get_frame(self) -> Optional[np.ndarray]:
        if self.pipeline is None: return None
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            color = frames.get_color_frame()
            if not color: return None
            return np.asanyarray(color.get_data())
        except Exception:
            return None

    def stop(self):
        if self.pipeline:
            try:
                self.pipeline.stop()
                print("[CAM] 相机已关闭")
            except Exception:
                pass

# ── 信号 / 飞书 / TTS / 工作流 ──────────────────────────────────────
def enqueue_command(cmd: str) -> bool:
    try:
        entry = {"ts": time.time(), "cmd": cmd, "raw": cmd, "source": "emotion_detect"}
        os.makedirs(os.path.dirname(QUEUE_FILE) or ".", exist_ok=True)
        with open(QUEUE_FILE, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return True
    except Exception as exc:
        print(f"[QUEUE ERROR] {exc}")
        return False

def _read_feishu_env() -> dict:
    env_path = os.path.expanduser("~/.oc_arm/feishu_alarm.env")
    if not os.path.exists(env_path): return {}
    env = {}
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            k, _, v = line.partition("=")
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"): v = v[1:-1]
            env[k.strip()] = v
    return env

def send_feishu_text(text: str) -> bool:
    env = _read_feishu_env()
    if not env.get("FEISHU_APP_ID") or not env.get("FEISHU_APP_SECRET"):
        return False
    try:
        import requests
        base = env.get("FEISHU_BASE_URL", "https://open.feishu.cn")
        r = requests.post(f"{base}/open-apis/auth/v3/tenant_access_token/internal",
                          json={"app_id": env["FEISHU_APP_ID"], "app_secret": env["FEISHU_APP_SECRET"]}, timeout=10)
        token = r.json().get("tenant_access_token", "")
        if not token: return False
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=utf-8"}
        payload = {"receive_id": env["FEISHU_ALARM_CHAT_ID"], "msg_type": "text",
                   "content": json.dumps({"text": text}, ensure_ascii=False)}
        r = requests.post(f"{base}/open-apis/im/v1/messages?receive_id_type={env.get('FEISHU_RECEIVE_ID_TYPE', 'chat_id')}",
                          headers=headers, json=payload, timeout=10)
        return r.status_code == 200
    except Exception:
        return False

def tts_via_voice_module(name: str) -> None:
    fifo = "/tmp/oc_voice_tts.fifo"
    if not os.path.exists(fifo): return
    try:
        with open(fifo, "w") as f:
            f.write(name + "\n")
            f.flush()
    except Exception:
        pass

def tts_spdsay(text: str) -> None:
    try:
        subprocess.run(["spd-say", "-l", "zh", text], timeout=10)
    except Exception:
        pass

def signal_xigua(frame_bgr: np.ndarray, boxes: np.ndarray, face_crop: Optional[np.ndarray] = None) -> None:
    full_path, crop_path = save_sad_image(frame_bgr, boxes, face_crop)
    print(f"[SAD] 图片已保存: {full_path}")
    print("\n===== [SAD] 检测到不开心，启动工作流 =====")

    feishu_text = "😢 你看起来不太开心，\n要不要喝杯，解解闷？\n小瓜陪着你哦🫂"
    send_feishu_text(feishu_text)

    tts_via_voice_module("emotion_unhappy")
    enqueue_command("sad_detected")

    workflow_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_run_shakebottle.sh")
    subprocess.Popen(["/bin/bash", workflow_script],
                     stdout=open("/tmp/emotion_workflow.log", "a"),
                     stderr=subprocess.STDOUT, start_new_session=True)
    print("[SAD] 飞书+语音已发送，摇瓶子工作流已在后台启动")

# ── 参数解析 ─────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="RDK BPU 3s 表情轮询 — YOLOv8 HBM 加速 + sad 触发")
    p.add_argument("--face-model", default="./weights/yolo11n_person_distance_quantized.hbm") 
    p.add_argument("--fer-model", default="./weights/facial_expression_recognition_mobilefacenet_2022july.onnx")
    p.add_argument("--camera-serial", default=CAMERA_SERIAL)
    p.add_argument("--input-size", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.7)       
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--max-det", type=int, default=1)         
    p.add_argument("--sad-frame-thres", type=float, default=0.75)
    p.add_argument("--sad-ratio-thres", type=float, default=0.50)
    p.add_argument("--sad-mean-thres", type=float, default=0.50)
    p.add_argument("--interval", type=float, default=2.0)
    p.add_argument("--smooth", type=int, default=3)
    p.add_argument("--debug", action="store_true", help="开启调试模式，每次识别到有效人脸都会保存带有标注框的完整画面到 debug_frames 文件夹")
    return p.parse_args()

# ── 主函数 ──────────────────────────────────────────────────────────
def main():
    args = parse_args()

    if not os.path.exists(args.face_model):
        print(f"[ERROR] HBM 人脸模型不存在: {args.face_model}", file=sys.stderr)
        return 2
    if not os.path.exists(args.fer_model):
        print(f"[ERROR] FER 模型不存在: {args.fer_model}", file=sys.stderr)
        return 2

    print(f"[INFO] 启动相机 (serial={args.camera_serial})...")
    camera = RealsenseCamera(args.camera_serial)
    if not camera.start():
        print("[ERROR] 相机启动失败", file=sys.stderr)
        return 2

    print("[INFO] 加载模型...")
    try:
        face_det = HBMYoloFace(args.face_model, args.input_size, args.conf, args.iou, args.max_det)
    except Exception as e:
        print(f"[ERROR] 加载 HBM 模型失败: {e}", file=sys.stderr)
        return 2
    
    fer = MobileFaceNetFERONNX(args.fer_model)

    sad_hist: deque = deque(maxlen=max(1, args.smooth))
    frame_count = 0
    started_at = datetime.now().isoformat(timespec="seconds")
    print(f"[INFO] 启动时间: {started_at}, 检测间隔={args.interval}s")

    if args.debug:
        print("[DEBUG] 调试模式已开启！将保存带标注的完整调试图片。")

    def shutdown():
        print("\n[INFO] 关闭中...")
        camera.stop()
        cv2.destroyAllWindows()

    import atexit
    atexit.register(shutdown)

    try:
        while True:
            with np.errstate(all='ignore'):
                frame = camera.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            frame_count += 1
            boxes, scores, yolo_infer_ms = face_det.detect(frame)

            label = "no_face"
            sad_score = 0.0
            emo_conf = 0.0
            face_crop = None
            fer_infer_ms = 0.0

            # 提取有效人脸并处理
            if len(boxes) > 0:
                # 取置信度最高的那个
                best_idx = int(np.argmax(scores))
                box = boxes[best_idx]
                
                x1, y1, x2, y2 = map(int, box)
                h, w = frame.shape[:2]
                
                box_area = (x2 - x1) * (y2 - y1)
                box_w = x2 - x1
                box_h = y2 - y1
                aspect_ratio = box_w / (box_h + 1e-6)
                
                # ===== 🛡️ 空间防呆过滤（继续保留，防假框非常有效） =====
                # 面积限制 + 人的长宽比 + 框顶不能低于画面 70%（防底部文字/衣服假框）
                if box_area > (h * w * 0.02) and box_area < (h * w * 0.8) and \
                   0.4 < aspect_ratio < 1.6 and y1 < h * 0.7:
                   
                    try:
                        # 直接使用 YOLO 框裁切
                        face = crop_face_112(frame, box, expand=0.10)
                        
                        if face is not None:
                            face_std = np.std(face)
                            if face_std > 25.0:  # 纯色过滤（防天花板假图）
                                face_crop = face
                                label, emo_conf, probs, fer_infer_ms = fer.predict(face)
                                sad_score = float(probs[SAD_INDEX])
                                probs_list = {EMOTIONS[i]: float(probs[i]) for i in range(len(EMOTIONS))}
                                
                                # debug 模式下保存带标注的整帧大图
                                if args.debug:
                                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                                    debug_frame = frame.copy()
                                    
                                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                    
                                    debug_label = f"{label} {emo_conf:.2f}"
                                    cv2.putText(debug_frame, debug_label, (x1, y1-10), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                    
                                    yolo_label = f"YOLO: {scores[best_idx]:.2f}"
                                    cv2.putText(debug_frame, yolo_label, (x1, y2+20), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                    
                                    debug_file = os.path.join(DEBUG_FRAME_DIR, f"{ts}_{label}_{emo_conf:.2f}.jpg")
                                    cv2.imwrite(debug_file, debug_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    except Exception:
                        pass

            sad_hist.append(sad_score)

            # 缓存本次检测结果
            cache_entry = {
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                "frame": frame_count,
                "label": label,
                "confidence": round(emo_conf, 4),
                "sad_score": round(sad_score, 4),
                "face_count": len(boxes),
                "inference_time_ms": {"yolo": round(yolo_infer_ms, 1), "fer": round(fer_infer_ms, 1)}
            }
            save_emotion_cache(cache_entry)

            if frame_count % 1 == 0:
                sad_avg = float(np.mean(sad_hist)) if sad_hist else 0.0
                print(f"[STATUS] frame#{frame_count} label={label} conf={emo_conf:.2f} sad_avg={sad_avg:.2f} "
                      f"yolo={yolo_infer_ms:.3f}ms fer={fer_infer_ms:.3f}ms", flush=True)

            # ==========================================================
            # 🎯 判定修改：由“滑动平均触发”改为“单帧即时阈值触发”
            # ==========================================================
            if label != "no_face" and sad_score >= args.sad_frame_thres:
                print(f"[SAD] 单帧 sad_score={sad_score:.2f} >= {args.sad_frame_thres}  → 立即发送信号并退出")
                signal_xigua(frame, boxes, face_crop)
                shutdown()
                return 0

            # 如果没触发 sad，则按照检测间隔正常等待下一帧
            time.sleep(args.interval)
            # ==========================================================

    except KeyboardInterrupt:
        print("\n[INFO] 用户中断")
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
    finally:
        shutdown()

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
