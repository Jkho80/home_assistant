#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenClaw / 小瓜语音交互模块串口桥接脚本（协议表对应修正版）

功能：
1. 监听语音模块串口，解析 5 字节协议帧：AA 55 TYPE ID FB
2. 打印模块识别到的命令，后续可映射到 oc_armctl 或其他脚本
3. 向语音模块发送被动播报帧，用于 TTS 播报
4. 默认只监听和打印，不执行危险动作；需要执行动作时显式加 --execute

依赖：
    pip install pyserial

典型用法：
    python3 oc_voice_uart_bridge.py monitor
    python3 oc_voice_uart_bridge.py monitor --port /dev/ttyUSB0
    python3 oc_voice_uart_bridge.py monitor --execute
    python3 oc_voice_uart_bridge.py tts task_running_safe
    python3 oc_voice_uart_bridge.py tts --frame "AA 55 FF A2 FB"
    python3 oc_voice_uart_bridge.py list

python3 oc_voice_uart_bridge_checked.py monitor --port /dev/ttyUSB0 --execute

"""

from __future__ import annotations

import argparse
import glob
import json
import os
import signal
import sys
import threading
import time
import urllib.request
import urllib.error
from datetime import datetime
from typing import Dict, Optional, Tuple

try:
    import serial
except ImportError:
    print("[ERROR] 缺少 pyserial，请先执行：pip install pyserial", file=sys.stderr)
    raise


BAUDRATE = 115200
FRAME_LEN = 5
FRAME_HEAD = bytes([0xAA, 0x55])
FRAME_END = 0xFB
TOKEN = os.environ.get("OPENCLAW_TOKEN", "")
if not TOKEN:
    print("[WARN] 未设置 OPENCLAW_TOKEN 环境变量", file=sys.stderr)
TYPE_COMMAND = 0x00   # 语音识别命令词
TYPE_TTS = 0xFF       # 被动播报/TTS


# =========================
# 命令词映射
# 若 Excel 中某个 ID 与这里不一致，优先改这里。
# =========================
COMMAND_MAP: Dict[int, Dict[str, object]] = {
    0x70: {
        "phrase": '帮我整理桌面',
        "action_key": 'smart_desktop_start',
        "description": '开始桌面整理/演示任务',
        "cmd": ['/home/sunrise/bin/oc_armctl', 'smart', 'start'],
    },
    0x71: {
        "phrase": '开始整理桌面',
        "action_key": 'smart_desktop_start',
        "description": '开始桌面整理/演示任务',
        "cmd": ['/home/sunrise/bin/oc_armctl', 'smart', 'start'],
    },
    0x72: {
        "phrase": '下面开始演示',
        "action_key": 'smart_desktop_start',
        "description": '开始桌面整理/演示任务',
        "cmd": ['/home/sunrise/bin/oc_armctl', 'smart', 'start'],
    },
    0x73: {
        "phrase": '停止当前任务',
        "action_key": 'task_stop',
        "description": '停止当前 smart 任务',
        "cmd": ['/home/sunrise/bin/oc_armctl', 'smart', 'stop'],
    },
    0x74: {
        "phrase": '立即停止任务',
        "action_key": 'task_stop',
        "description": '停止当前 smart 任务',
        "cmd": ['/home/sunrise/bin/oc_armctl', 'smart', 'stop'],
    },
    0x75: {
        "phrase": '开启手柄模式',
        "action_key": 'gamepad_start',
        "description": '进入手柄控制模式',
        "cmd": ['/home/sunrise/bin/oc_armctl', 'gamepad', 'start'],
    },
    0x76: {
        "phrase": '进入手柄模式',
        "action_key": 'gamepad_start',
        "description": '进入手柄控制模式',
        "cmd": ['/home/sunrise/bin/oc_armctl', 'gamepad', 'start'],
    },
    0x77: {
        "phrase": '退出手柄模式',
        "action_key": 'gamepad_stop',
        "description": '退出手柄控制模式',
        "cmd": ['/home/sunrise/bin/oc_armctl', 'gamepad', 'stop'],
    },
    0x78: {
        "phrase": '关闭手柄模式',
        "action_key": 'gamepad_stop',
        "description": '退出手柄控制模式',
        "cmd": ['/home/sunrise/bin/oc_armctl', 'gamepad', 'stop'],
    },
    0x79: {
        "phrase": '双臂回零',
        "action_key": 'dual_arm_reset',
        "description": '双臂回零；注意 oc_armctl 实际命令是 arm reset，不是顶层 reset',
        "cmd": ['/home/sunrise/bin/oc_armctl', 'arm', 'reset'],
    },
    0x7A: {
        "phrase": '机械臂回零',
        "action_key": 'dual_arm_reset',
        "description": '双臂回零；注意 oc_armctl 实际命令是 arm reset，不是顶层 reset',
        "cmd": ['/home/sunrise/bin/oc_armctl', 'arm', 'reset'],
    },
    0x7B: {
        "phrase": '启动相机服务',
        "action_key": 'camera_start',
        "description": '启动相机服务',
        "cmd": ['/home/sunrise/bin/oc_armctl', 'camera', 'start'],
    },
    0x7C: {
        "phrase": '停止相机服务',
        "action_key": 'camera_stop',
        "description": '停止相机服务',
        "cmd": ['/home/sunrise/bin/oc_armctl', 'camera', 'stop'],
    },
    0x7D: {
        "phrase": '重启相机服务',
        "action_key": 'camera_restart',
        "description": '重启相机服务',
        "cmd": ['/home/sunrise/bin/oc_armctl', 'camera', 'restart'],
    },
    0x7E: {
        "phrase": '启动机械臂服务',
        "action_key": 'arm_start',
        "description": '启动机械臂 ROS 服务',
        "cmd": ['/home/sunrise/bin/oc_armctl', 'arm', 'start'],
    },
    0x7F: {
        "phrase": '停止机械臂服务',
        "action_key": 'arm_stop',
        "description": '停止机械臂 ROS 服务',
        "cmd": ['/home/sunrise/bin/oc_armctl', 'arm', 'stop'],
    },
    0x80: {
        "phrase": '重启机械臂服务',
        "action_key": 'arm_restart',
        "description": '重启机械臂 ROS 服务',
        "cmd": ['/home/sunrise/bin/oc_armctl', 'arm', 'restart'],
    },
    0x81: {
        "phrase": '启动模型服务',
        "action_key": 'server_start',
        "description": '启动 HoloBrain / 模型服务端',
        "cmd": ['/home/sunrise/bin/oc_armctl', 'server', 'start'],
    },
    0x82: {
        "phrase": '停止模型服务',
        "action_key": 'server_stop',
        "description": '停止 HoloBrain / 模型服务端',
        "cmd": ['/home/sunrise/bin/oc_armctl', 'server', 'stop'],
    },
    0x83: {
        "phrase": '重启模型服务',
        "action_key": 'server_restart',
        "description": '重启 HoloBrain / 模型服务端',
        "cmd": ['/home/sunrise/bin/oc_armctl', 'server', 'restart'],
    },
    0x84: {
        "phrase": '启动智能服务',
        "action_key": 'smart_start',
        "description": '启动 smart client',
        "cmd": ['/home/sunrise/bin/oc_armctl', 'smart', 'start'],
    },
    0x85: {
        "phrase": '停止智能服务',
        "action_key": 'smart_stop',
        "description": '停止 smart client',
        "cmd": ['/home/sunrise/bin/oc_armctl', 'smart', 'stop'],
    },
    0x86: {
        "phrase": '重启智能服务',
        "action_key": 'smart_restart',
        "description": '重启 smart client',
        "cmd": ['/home/sunrise/bin/oc_armctl', 'smart', 'restart'],
    },
    0x87: {
        "phrase": '查询系统状态',
        "action_key": 'system_status',
        "description": '快速状态检查；oc_armctl 无顶层 status，使用 fast',
        "cmd": ['/home/sunrise/bin/oc_armctl', 'fast'],
    },
    0x88: {
        "phrase": '检查系统健康',
        "action_key": 'system_health',
        "description": '完整健康检查',
        "cmd": ['/home/sunrise/bin/oc_armctl', 'health'],
    },
    0x89: {
        "phrase": '紧急停止',
        "action_key": 'emergency_stop',
        "description": '语音紧急停止；当前先映射为 smart stop',
        "cmd": ['/home/sunrise/bin/oc_armctl', 'smart', 'stop'],
    },
    0x8A: {
        "phrase": '停止运行',
        "action_key": 'emergency_stop',
        "description": '语音紧急停止；当前先映射为 smart stop',
        "cmd": ['/home/sunrise/bin/oc_armctl', 'smart', 'stop'],
    },
    0x8B: {
        "phrase": '启动安全检测',
        "action_key": 'safety_watchdog_start',
        "description": '启动严重错误 watchdog 服务',
        "cmd": ['/home/sunrise/bin/oc_armctl', 'error', 'start'],
    },
    0x8C: {
        "phrase": '停止安全检测',
        "action_key": 'safety_watchdog_stop',
        "description": '停止严重错误 watchdog 服务',
        "cmd": ['/home/sunrise/bin/oc_armctl', 'error', 'stop'],
    },
    0x8D: {
        "phrase": '重启安全检测',
        "action_key": 'safety_watchdog_restart',
        "description": '重启严重错误 watchdog 服务',
        "cmd": ['/home/sunrise/bin/oc_armctl', 'error', 'restart'],
    },
    0x8E: {
        "phrase": '查询手柄状态',
        "action_key": 'gamepad_status',
        "description": '查询手柄模式状态',
        "cmd": ['/home/sunrise/bin/oc_armctl', 'gamepad', 'status'],
    },
    0x8F: {
        "phrase": '启动表情识别',
        "action_key": 'fer_start',
        "description": '启动表情识别；外部脚本路径待确认，默认只打印不执行',
        "cmd": None,
    },
    0x90: {
        "phrase": '停止表情识别',
        "action_key": 'fer_stop',
        "description": '停止表情识别；外部脚本路径待确认，默认只打印不执行',
        "cmd": None,
    },
    0x91: {
        "phrase": '检查运行日志',
        "action_key": 'system_logs',
        "description": '查看 oc_armctl 汇总日志',
        "cmd": ['/home/sunrise/bin/oc_armctl', 'logs'],
    },
}


# =========================
# 被动 TTS 播报映射
# =========================
TTS_MAP: Dict[str, Dict[str, object]] = {
    'emotion_unhappy': {"text": '您似乎不太开心', "id": 0xA0},
    'distance_too_close': {"text": '检测到距离过近', "id": 0xA1},
    'task_running_safe': {"text": '任务进行中，请注意安全', "id": 0xA2},
    'gamepad_started': {"text": '已进入手柄模式', "id": 0xA3},
    'gamepad_stopped': {"text": '已退出手柄模式', "id": 0xA4},
    'arms_resetting': {"text": '双臂正在回零', "id": 0xA5},
    'severe_error': {"text": '检测到严重错误', "id": 0xA6},
    'power_restart_required': {"text": '请手动断电重启', "id": 0xA7},
    'service_started': {"text": '服务启动成功', "id": 0xA8},
    'service_failed': {"text": '服务启动失败', "id": 0xA9},
    'device_connection_error': {"text": '连接异常，请检查设备', "id": 0xAA},
    'task_finished': {"text": '任务已完成', "id": 0xAB},
    'task_stopped': {"text": '任务已停止', "id": 0xAC},
    'system_ok': {"text": '系统状态正常', "id": 0xAD},
    'service_starting': {"text": '正在启动服务', "id": 0xAE},
    'safety_watchdog_started': {"text": '安全检测已开启', "id": 0xAF},
}


GATEWAY_CONFIG_CACHE: Optional[dict] = None


SESSION_KEY_CACHE: Optional[str] = None


TTS_FIFO = "/tmp/oc_voice_tts.fifo"

QUEUE_FILE = "/tmp/oc_voice_queue.jsonl"
TTS_EMOTION_UNHAPPY_FRAME = bytes([0xAA, 0x55, 0xFF, 0xA0, 0xFB])

RUNNING = True


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def hex_bytes(data: bytes) -> str:
    return data.hex(" ").upper()


def make_frame(frame_type: int, data_id: int) -> bytes:
    if not (0 <= frame_type <= 0xFF and 0 <= data_id <= 0xFF):
        raise ValueError("frame_type 和 data_id 必须是 0-255")
    return bytes([0xAA, 0x55, frame_type, data_id, 0xFB])


def parse_hex_frame(text: str) -> bytes:
    cleaned = text.replace("0x", "").replace("0X", "").replace(",", " ").replace("-", " ")
    parts = [p for p in cleaned.split() if p.strip()]
    data = bytes(int(p, 16) for p in parts)
    if len(data) != FRAME_LEN:
        raise ValueError(f"协议帧必须是 {FRAME_LEN} 字节，当前是 {len(data)} 字节：{hex_bytes(data)}")
    if data[:2] != FRAME_HEAD or data[-1] != FRAME_END:
        raise ValueError(f"协议帧格式错误，应为 AA 55 XX XX FB，当前：{hex_bytes(data)}")
    return data


def find_default_port() -> Optional[str]:
    candidates = []
    for pattern in ("/dev/ttyUSB*", "/dev/ttyACM*", "/dev/serial/by-id/*"):
        candidates.extend(glob.glob(pattern))
    candidates = sorted(candidates)
    return candidates[0] if candidates else None


def open_serial(port: Optional[str], baudrate: int, timeout: float) -> serial.Serial:
    if not port:
        port = find_default_port()
    if not port:
        raise RuntimeError("没有找到串口设备。请检查 ls /dev/ttyUSB*，或显式指定 --port /dev/voice")
    print(f"[INFO] opening serial: port={port}, baudrate={baudrate}")
    return serial.Serial(port=port, baudrate=baudrate, timeout=timeout)


def decode_frame(frame: bytes) -> Dict[str, object]:
    frame_type = frame[2]
    data_id = frame[3]

    result: Dict[str, object] = {
        "time": now_str(),
        "raw": hex_bytes(frame),
        "type": f"0x{frame_type:02X}",
        "id": f"0x{data_id:02X}",
        "kind": "unknown",
    }

    if frame_type == TYPE_COMMAND:
        item = COMMAND_MAP.get(data_id)
        result["kind"] = "command"
        if item:
            result.update({
                "phrase": item["phrase"],
                "action_key": item["action_key"],
                "description": item["description"],
            })
        else:
            result.update({
                "phrase": None,
                "action_key": None,
                "description": "未登记命令词 ID",
            })
    elif frame_type == TYPE_TTS:
        result["kind"] = "tts_or_broadcast"
        matched_key = None
        matched_item = None
        for key, item in TTS_MAP.items():
            if int(item["id"]) == data_id:
                matched_key = key
                matched_item = item
                break
        if matched_item:
            result.update({"tts_key": matched_key, "text": matched_item["text"]})
        else:
            result.update({"tts_key": None, "text": None, "description": "未登记播报语 ID"})
    else:
        result["description"] = "未知协议类型"

    return result


def _load_gateway_config() -> dict:
    """从 ~/.openclaw/openclaw.json 读取 gateway 配置（port, token）"""
    global GATEWAY_CONFIG_CACHE
    if GATEWAY_CONFIG_CACHE is not None:
        return GATEWAY_CONFIG_CACHE
    cfg_path = os.path.expanduser("~/.openclaw/openclaw.json")
    try:
        with open(cfg_path) as f:
            data = json.load(f)
        gw = data.get("gateway", {})
        GATEWAY_CONFIG_CACHE = {
            "port": gw.get("port", 18789),
            "token": gw.get("auth", {}).get("token", ""),
        }
        return GATEWAY_CONFIG_CACHE
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"[ERROR] 无法读取 gateway 配置 ({cfg_path}): {e}", file=sys.stderr)
        return {"port": 18789, "token": ""}


def send_to_session(phrase: str, session_key: str) -> bool:
    """通过 Gateway tool invoke 将语音命令推送到会话"""
    cfg = _load_gateway_config()
    token = TOKEN or cfg.get("token", "")
    if not token:
        print("[ERROR] 未找到 token", file=sys.stderr)
        return False

    url = f"http://127.0.0.1:{cfg['port']}/tools/invoke"
    body = json.dumps({
        "tool": "sessions_send",
        "args": {
            "sessionKey": session_key,
            "message": f"[语音] {phrase}"
        }
    }).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            print(f"[SEND] '[语音] {phrase}' -> 会话 {session_key} (status={resp.status})")
            return True
    except urllib.error.HTTPError as e:
        print(f"[ERROR] HTTP {e.code}: {e.read().decode()[:200]}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[ERROR] 发送失败: {e}", file=sys.stderr)
        return False
def send_to_session_async(phrase: str, session_keys: list) -> None:
    """使用 curl 调用 chat completions，异步发送语音命令到多个会话"""
    def _send():
        import subprocess
        cfg = _load_gateway_config()
        token = TOKEN or cfg.get("token", "")
        if not token:
            print("[ERROR] 未找到 token", file=sys.stderr)
            return
        body = json.dumps({
            "model": "openclaw/default",
            "messages": [{"role": "user", "content": phrase}],
        })
        for sk in session_keys:
            try:
                cmd = [
                    "curl", "-s", "--max-time", "3", "-X", "POST",
                    f"http://127.0.0.1:{cfg['port']}/v1/chat/completions",
                    "-H", "Content-Type: application/json",
                    "-H", f"Authorization: Bearer {token}",
                    "-H", "x-openclaw-session-key: " + sk,
                    "-d", body,
                ]
                subprocess.run(cmd, capture_output=True, timeout=5)
                print(f"[SEND] '{phrase}' -> 会话 {sk}")
            except Exception as e:
                print(f"[WARN] 发送 '{phrase}' -> {sk} 失败（非致命）: {e}", file=sys.stderr)
    threading.Thread(target=_send, daemon=True).start()

def execute_action(data_id: int, phrase: str, session_keys: list, dry_run: bool = False) -> int:
    """将语音命令以 user 消息发送到多个会话"""
    if dry_run:
        for sk in session_keys:
            print(f"[DRY-RUN] 将发送 '{phrase}' 到会话 {sk}")
        return 0

    if not phrase:
        print(f"[SKIP] 未识别的命令 ID 0x{data_id:02X}，不发送", file=sys.stderr)
        return 1

    if not session_keys:
        print("[ERROR] 未指定会话 key（使用 --session-key）", file=sys.stderr)
        return 1

    send_to_session_async(phrase, session_keys)
    return 0


def _tts_fifo_listener(ser: serial.Serial, stop_event: threading.Event) -> None:
    """后台线程：监听 TTS FIFO，收到请求后直接写帧到串口"""
    try:
        os.mkfifo(TTS_FIFO)
    except FileExistsError:
        pass

    while not stop_event.is_set():
        try:
            with open(TTS_FIFO, 'r') as fifo:
                line = fifo.readline().strip()
                if not line:
                    continue
                item = TTS_MAP.get(line)
                if item:
                    frame = make_frame(TYPE_TTS, int(item["id"]))
                    ser.write(frame)
                    ser.flush()
                    print(f"[TTS] {hex_bytes(frame)}  # {item.get('text', '')}")
                else:
                    print(f"[TTS] 未知 TTS 名称: {line}")
        except Exception as e:
            if not stop_event.is_set():
                print(f"[TTS] FIFO listener error: {e}")
                time.sleep(0.5)


def monitor(args: argparse.Namespace) -> None:
    global RUNNING

    ser = open_serial(args.port, args.baudrate, args.timeout)
    buffer = bytearray()
    last_event: Dict[Tuple[int, int], float] = {}

    print("[INFO] monitor started. Press Ctrl+C to exit.")
    print("[INFO] 默认只打印。若要执行动作，请使用 --execute。")
    print(f"[INFO] TTS FIFO: {TTS_FIFO} (echo 'tts_name' > {TTS_FIFO})")

    # 启动 TTS 监听线程
    stop_event = threading.Event()
    tts_thread = threading.Thread(
        target=_tts_fifo_listener, args=(ser, stop_event), daemon=True
    )
    tts_thread.start()

    try:
        while RUNNING:
            chunk = ser.read(args.read_size)
            if chunk:
                buffer.extend(chunk)

            if len(buffer) > 1024:
                print(f"[WARN] buffer too large, clear. tail={hex_bytes(bytes(buffer[-20:]))}")
                buffer.clear()

            while len(buffer) >= FRAME_LEN:
                head_idx = buffer.find(FRAME_HEAD)
                if head_idx < 0:
                    if buffer:
                        del buffer[:-1]
                    break

                if head_idx > 0:
                    noise = bytes(buffer[:head_idx])
                    if args.print_noise:
                        print(f"[NOISE] {hex_bytes(noise)}")
                    del buffer[:head_idx]

                if len(buffer) < FRAME_LEN:
                    break

                candidate = bytes(buffer[:FRAME_LEN])
                if candidate[-1] != FRAME_END:
                    if args.print_noise:
                        print(f"[BAD] {hex_bytes(candidate)}")
                    del buffer[0]
                    continue

                del buffer[:FRAME_LEN]

                decoded = decode_frame(candidate)
                frame_type = candidate[2]
                data_id = candidate[3]
                event_key = (frame_type, data_id)
                now = time.time()

                if args.debounce > 0:
                    prev = last_event.get(event_key, 0)
                    if now - prev < args.debounce:
                        print(f"[SKIP] debounce {args.debounce}s: {hex_bytes(candidate)}")
                        continue
                    last_event[event_key] = now

                if args.json:
                    print(json.dumps(decoded, ensure_ascii=False))
                else:
                    print(f"\n[{decoded['time']}] RX {decoded['raw']}")
                    print(f"  kind={decoded.get('kind')} type={decoded.get('type')} id={decoded.get('id')}")
                    if decoded.get("phrase"):
                        print(f"  phrase={decoded.get('phrase')}")
                    if decoded.get("action_key"):
                        print(f"  action_key={decoded.get('action_key')}")
                    if decoded.get("text"):
                        print(f"  text={decoded.get('text')}")
                    if decoded.get("description"):
                        print(f"  description={decoded.get('description')}")

                if args.execute and frame_type == TYPE_COMMAND:
                    phrase = decoded.get("phrase") or ""
                    execute_action(data_id, phrase=phrase, session_keys=args.session_key, dry_run=args.dry_run)

            if not chunk:
                time.sleep(0.01)

    finally:
        stop_event.set()
        ser.close()
        print("[INFO] serial closed")


def send_tts(args: argparse.Namespace) -> None:
    """通过直接打开串口发送 TTS（独立模式，不受 monitor 影响）"""
    ser = open_serial(args.port, args.baudrate, args.timeout)

    if args.frame:
        frame = parse_hex_frame(args.frame)
        text = "(custom frame)"
    else:
        item = TTS_MAP.get(args.name)
        if not item:
            print(f"[ERROR] 未找到 TTS 名称：{args.name}")
            print("可用名称：")
            for key in TTS_MAP:
                print(f"  {key}")
            sys.exit(2)
        frame = make_frame(TYPE_TTS, int(item["id"]))
        text = str(item["text"])

    try:
        ser.write(frame)
        ser.flush()
        print(f"[TX] {hex_bytes(frame)}  # {text}")
    finally:
        ser.close()


def send_tts_request(args: argparse.Namespace) -> None:
    """通过 FIFO 向 monitor 发送 TTS 请求"""
    name = args.name
    if name not in TTS_MAP:
        print(f"[ERROR] 未找到 TTS 名称：{name}")
        print("可用名称：")
        for key in TTS_MAP:
            print(f"  {key}")
        sys.exit(2)

    try:
        # 确保 FIFO 存在
        try:
            os.mkfifo(TTS_FIFO)
        except FileExistsError:
            pass

        with open(TTS_FIFO, 'w') as fifo:
            fifo.write(name + '\n')
            fifo.flush()
        print(f"[TTS-REQ] '{name}' -> FIFO {TTS_FIFO}")
    except Exception as e:
        print(f"[ERROR] 写入 FIFO 失败: {e}", file=sys.stderr)
        print("[HINT] 确保 monitor 正在运行（python3 oc_voice_session.py monitor）")
        sys.exit(1)


def list_items(_: argparse.Namespace) -> None:
    print("\n[COMMAND_MAP]")
    for data_id, item in sorted(COMMAND_MAP.items()):
        frame = make_frame(TYPE_COMMAND, data_id)
        cmd = item.get("cmd")
        cmd_text = " ".join(cmd) if isinstance(cmd, list) else "-"
        print(f"  0x{data_id:02X}  {hex_bytes(frame)}  {item['phrase']}  -> {item['action_key']}  -> {cmd_text}")

    print("\n[TTS_MAP]")
    for key, item in sorted(TTS_MAP.items(), key=lambda kv: int(kv[1]["id"])):
        data_id = int(item["id"])
        frame = make_frame(TYPE_TTS, data_id)
        print(f"  {key:24s}  0x{data_id:02X}  {hex_bytes(frame)}  {item['text']}")


def handle_signal(signum, frame) -> None:  # type: ignore[no-untyped-def]
    global RUNNING
    RUNNING = False
    print(f"\n[INFO] received signal {signum}, exiting...")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenClaw 小瓜语音模块串口桥接脚本")
    sub = parser.add_subparsers(dest="subcmd", required=True)

    def add_serial_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--port", default="/dev/voice", help="串口设备，例如 /dev/ttyUSB0；不填则自动寻找")
        p.add_argument("--baudrate", type=int, default=BAUDRATE, help="波特率，默认 115200")
        p.add_argument("--timeout", type=float, default=0.05, help="串口 read 超时，默认 0.05 秒")

    p_monitor = sub.add_parser("monitor", help="监听串口，解析语音模块上报帧")
    add_serial_args(p_monitor)
    p_monitor.add_argument("--read-size", type=int, default=64, help="每次读取字节数")
    p_monitor.add_argument("--execute", action="store_true", help="识别到命令后以 user 身份发送到当前会话")
    default_session = os.environ.get("OPENCLAW_SESSION_KEY",
                                    "agent:main:feishu:group:oc_e0d46e497e4fddebe4706f2e8f206225")
    p_monitor.add_argument("--session-key", action="append", default=[default_session],
                           help="目标会话 key，可多次指定（与 --execute 配合使用）")
    p_monitor.add_argument("--dry-run", action="store_true", help="只打印将要发送的内容，不真正发送")
    p_monitor.add_argument("--exec-timeout", type=int, default=20, help="动作执行超时时间（当前未使用）")
    p_monitor.add_argument("--debounce", type=float, default=1.0, help="同一协议帧去抖时间，默认 1 秒；0 表示关闭")
    p_monitor.add_argument("--json", action="store_true", help="按 JSON Lines 输出，便于被其他程序读取")
    p_monitor.add_argument("--print-noise", action="store_true", help="打印非协议噪声数据")
    p_monitor.set_defaults(func=monitor)

    p_tts = sub.add_parser("tts", help="向语音模块发送被动播报帧")
    add_serial_args(p_tts)
    p_tts.add_argument("name", nargs="?", help="TTS 名称，例如 task_running_safe")
    p_tts.add_argument("--frame", default=None, help='直接发送原始 5 字节协议，例如 "AA 55 FF A2 FB"')
    p_tts.set_defaults(func=send_tts)

    p_tts_req = sub.add_parser("tts-req", help="通过 FIFO 向 monitor 发送 TTS 请求（需 monitor 正在运行）")
    p_tts_req.add_argument("name", help="TTS 名称，例如 service_started")
    p_tts_req.set_defaults(func=send_tts_request)

    p_list = sub.add_parser("list", help="列出当前命令词/TTS 映射")
    p_list.set_defaults(func=list_items)

    return parser


def main() -> None:
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    parser = build_parser()
    args = parser.parse_args()

    if args.subcmd == "tts" and not args.frame and not args.name:
        parser.error("tts 子命令需要提供 name 或 --frame")

    args.func(args)


if __name__ == "__main__":
    main()
