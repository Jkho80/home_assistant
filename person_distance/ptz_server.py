#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ptz_server.py — PTZ Control Server (基于 Unix Socket + 队列的串行化云台控制)
======================================================================

架构：
  ptz_server.py  (独立守护进程)
      │
      ├── Unix Socket (/tmp/ptz_server.sock) ← 客户端（YOLO 检测/手动控制）
      │
      ├── PanTiltController (yuntai.py) ← 底层云台控制
      │
      └── 内部工作队列 ← 所有命令通过队列串行执行，消除竞态

解决的问题：
  - 云台到达位置与预期不一致（开环控制累积误差）
  - 多个 start_move_pan 同时发出，云台指令冲突
  - 上下限控制不一致

协议：JSON over Unix Socket
  请求: {"cmd": "...", ...}
  响应: {"ok": true/false, "pan": float, "tilt": float, ...}

命令列表：
  move_to      阻塞  {"cmd":"move_to", "pan":90, "tilt":45}
  move_pan     非阻塞 {"cmd":"move_pan", "pan":90}
  move_tilt    非阻塞 {"cmd":"move_tilt", "tilt":45}
  stop         立即   {"cmd":"stop"}             → 停止当前运动
  clear        立即   {"cmd":"clear"}            → 清空队列 + 停止运动
  get_position 查询   {"cmd":"get_position"}     → 返回当前位置
  get_status   查询   {"cmd":"get_status"}       → 返回位置+队列状态
  reset_pan    校准   {"cmd":"reset_pan", "pan":90}
  reset_tilt   校准   {"cmd":"reset_tilt", "tilt":45}
  set_limits   配置   {"cmd":"set_limits", "pan_min":0, "pan_max":180, "tilt_min":0, "tilt_max":90}
  home         非阻塞 {"cmd":"home"}             → 回到初始位置
  shutdown     立即   {"cmd":"shutdown"}         → 停止服务器

管理系统：
  python3 ptz_server.py start   → 后台启动
  python3 ptz_server.py stop    → 停止
  python3 ptz_server.py restart → 重启
  python3 ptz_server.py status  → 查看状态
"""

import argparse
import json
import os
import signal
import socket
import sys
import threading
import time
import logging
from queue import Queue, Empty
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger("ptz_server")

# ---------- 配置 ----------
SOCKET_PATH = "/tmp/ptz_server.sock"
PID_FILE = "/tmp/ptz_server.pid"
DEFAULT_PORT = "/dev/ptz"
DEFAULT_BAUDRATE = 9600
DEFAULT_INITIAL_PAN = 90.0
DEFAULT_INITIAL_TILT = 45.0

# ============================================================
# PanTiltControllerWrapper — 云台底层控制（带完善的状态管理）
# ============================================================

class PanTiltControllerWrapper:
    """
    包装 PanTiltController，提供序列化安全的运动控制。
    所有运动都是阻塞的（在独立工作线程中执行）。
    """

    PAN_MIN, PAN_MAX = 0.0, 180.0
    TILT_MIN, TILT_MAX = 0.0, 90.0

    def __init__(self, port=DEFAULT_PORT, baudrate=DEFAULT_BAUDRATE,
                 initial_pan=DEFAULT_INITIAL_PAN, initial_tilt=DEFAULT_INITIAL_TILT):
        from yuntai import PanTiltController
        self.pt = PanTiltController(
            port=port, baudrate=baudrate,
            initial_pan=initial_pan, initial_tilt=initial_tilt
        )
        self.pan = self.pt.pan
        self.tilt = self.pt.tilt
        self.port = port
        self.baudrate = baudrate
        self.initial_pan = initial_pan
        self.initial_tilt = initial_tilt
        LOGGER.info(f"✅ PanTiltControllerWrapper 初始化: pan={self.pan:.1f}°, tilt={self.tilt:.1f}°")

    def set_pan(self, target_pan):
        """阻塞式水平运动（完整执行后返回）"""
        target_pan = self._clamp(target_pan, self.PAN_MIN, self.PAN_MAX)
        diff = target_pan - self.pan
        if abs(diff) < 1.0:
            self.pan = target_pan
            LOGGER.info(f"  pan 已在 {target_pan:.1f}°，无需移动")
            return
        direction = "right" if diff > 0 else "left"
        dur = abs(diff) / self.pt.PAN_SPEED
        LOGGER.info(f"  pan {self.pan:.1f}° → {target_pan:.1f}° (diff={diff:+.1f}°, {direction}, {dur:.1f}s)")
        self.pt._send(self.pt.CMD_RIGHT if diff > 0 else self.pt.CMD_LEFT)
        time.sleep(dur)
        self.pt._send(self.pt.CMD_STOP)
        self.pan = target_pan
        LOGGER.info(f"  ✅ pan 到位: {self.pan:.1f}°")

    def set_tilt(self, target_tilt):
        """阻塞式俯仰运动（完整执行后返回）"""
        target_tilt = self._clamp(target_tilt, self.TILT_MIN, self.TILT_MAX)
        diff = target_tilt - self.tilt
        if abs(diff) < 1.0:
            self.tilt = target_tilt
            LOGGER.info(f"  tilt 已在 {target_tilt:.1f}°，无需移动")
            return
        direction = "up" if diff > 0 else "down"
        dur = abs(diff) / self.pt.TILT_SPEED
        LOGGER.info(f"  tilt {self.tilt:.1f}° → {target_tilt:.1f}° (diff={diff:+.1f}°, {direction}, {dur:.1f}s)")
        self.pt._send(self.pt.CMD_UP if diff > 0 else self.pt.CMD_DOWN)
        time.sleep(dur)
        self.pt._send(self.pt.CMD_STOP)
        self.tilt = target_tilt
        LOGGER.info(f"  ✅ tilt 到位: {self.tilt:.1f}°")

    def move_to(self, target_pan, target_tilt):
        """移动到指定 pan 和 tilt（先 pan 后 tilt）"""
        self.set_pan(target_pan)
        self.set_tilt(target_tilt)

    def stop(self):
        """立即停止"""
        self.pt._send(self.pt.CMD_STOP)
        LOGGER.info("  ⏹️ 云台已停止")

    def reset_pan(self, actual_pan):
        """校准当前物理 pan 角度（不移动云台）"""
        self.pan = self._clamp(actual_pan, self.PAN_MIN, self.PAN_MAX)
        LOGGER.info(f"  📐 pan 校准: {self.pan:.1f}°")

    def reset_tilt(self, actual_tilt):
        """校准当前物理 tilt 角度（不移动云台）"""
        self.tilt = self._clamp(actual_tilt, self.TILT_MIN, self.TILT_MAX)
        LOGGER.info(f"  📐 tilt 校准: {self.tilt:.1f}°")

    def set_limits(self, pan_min=None, pan_max=None, tilt_min=None, tilt_max=None):
        if pan_min is not None: self.PAN_MIN = pan_min
        if pan_max is not None: self.PAN_MAX = pan_max
        if tilt_min is not None: self.TILT_MIN = tilt_min
        if tilt_max is not None: self.TILT_MAX = tilt_max
        LOGGER.info(f"  限制更新: pan [{self.PAN_MIN}, {self.PAN_MAX}], tilt [{self.TILT_MIN}, {self.TILT_MAX}]")

    def get_state(self):
        return {
            "pan": round(self.pan, 1),
            "tilt": round(self.tilt, 1),
            "pan_min": self.PAN_MIN,
            "pan_max": self.PAN_MAX,
            "tilt_min": self.TILT_MIN,
            "tilt_max": self.TILT_MAX,
        }

    def close(self):
        try:
            self.pt.close()
        except Exception:
            pass

    @staticmethod
    def _clamp(value, min_val, max_val):
        return max(min_val, min(max_val, value))


# ============================================================
# 命令队列处理器
# ============================================================

class PtzCommand:
    """一条待执行的云台命令"""

    def __init__(self, cmd_type, params, response_queue):
        self.cmd_type = cmd_type  # "move_pan" | "move_tilt" | "move_to" | "stop" | ...
        self.params = params
        self.response_queue = response_queue  # 用于返回结果的 Queue


class CommandWorker(threading.Thread):
    """工作线程：从队列取命令并串行执行"""

    def __init__(self, ptz, cmd_queue):
        super().__init__(daemon=True, name="ptz-worker")
        self.ptz = ptz
        self.cmd_queue = cmd_queue
        self._stop_flag = threading.Event()
        self._skip_current = threading.Event()

    def run(self):
        LOGGER.info("🔧 PTZ 工作线程已启动")
        while not self._stop_flag.is_set():
            try:
                cmd = self.cmd_queue.get(timeout=1.0)
            except Empty:
                continue

            if cmd is None:
                continue

            if self._skip_current.is_set():
                self._skip_current.clear()
                cmd.response_queue.put({"ok": True, "msg": "skipped",
                                        **self.ptz.get_state()})
                continue

            try:
                result = self._execute(cmd)
            except Exception as e:
                LOGGER.error(f"执行命令失败: {e}")
                result = {"ok": False, "error": str(e), **self.ptz.get_state()}

            cmd.response_queue.put(result)
            self.cmd_queue.task_done()

        LOGGER.info("🛑 PTZ 工作线程已退出")

    def _execute(self, cmd):
        cmd_type = cmd.cmd_type
        params = cmd.params
        ptz = self.ptz

        if cmd_type == "move_pan":
            target = params.get("pan", ptz.pan)
            ptz.set_pan(target)
            return {"ok": True, **ptz.get_state()}

        elif cmd_type == "move_tilt":
            target = params.get("tilt", ptz.tilt)
            ptz.set_tilt(target)
            return {"ok": True, **ptz.get_state()}

        elif cmd_type == "move_to":
            pan = params.get("pan", ptz.pan)
            tilt = params.get("tilt", ptz.tilt)
            ptz.move_to(pan, tilt)
            return {"ok": True, **ptz.get_state()}

        elif cmd_type == "home":
            ptz.move_to(ptz.initial_pan, ptz.initial_tilt)
            return {"ok": True, **ptz.get_state()}

        elif cmd_type == "stop":
            ptz.stop()
            return {"ok": True, **ptz.get_state()}

        elif cmd_type == "reset_pan":
            ptz.reset_pan(params.get("pan", ptz.pan))
            return {"ok": True, **ptz.get_state()}

        elif cmd_type == "reset_tilt":
            ptz.reset_tilt(params.get("tilt", ptz.tilt))
            return {"ok": True, **ptz.get_state()}

        elif cmd_type == "set_limits":
            ptz.set_limits(
                pan_min=params.get("pan_min"),
                pan_max=params.get("pan_max"),
                tilt_min=params.get("tilt_min"),
                tilt_max=params.get("tilt_max"),
            )
            return {"ok": True, **ptz.get_state()}

        else:
            return {"ok": False, "error": f"未知命令: {cmd_type}", **ptz.get_state()}

    def skip_current(self):
        """跳过当前正在执行的命令"""
        self._skip_current.set()

    def stop(self):
        self._stop_flag.set()


# ============================================================
# PTZ Server
# ============================================================

class PtzServer:
    """PTZ 控制服务器（Unix Socket + 串行命令队列）"""

    def __init__(self, port=DEFAULT_PORT, baudrate=DEFAULT_BAUDRATE,
                 initial_pan=DEFAULT_INITIAL_PAN, initial_tilt=DEFAULT_INITIAL_TILT):
        self.ptz = PanTiltControllerWrapper(port, baudrate, initial_pan, initial_tilt)
        self.cmd_queue = Queue()
        self.worker = CommandWorker(self.ptz, self.cmd_queue)
        self.socket_path = SOCKET_PATH
        self.server_socket = None
        self._running = threading.Event()
        self._stop_signaled = threading.Event()

    def start(self):
        """启动服务器"""
        # 清理旧的 socket 文件
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError:
                LOGGER.warning(f"无法删除旧的 socket 文件: {self.socket_path}")

        # 创建 Unix socket
        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.bind(self.socket_path)
        self.server_socket.listen(10)
        os.chmod(self.socket_path, 0o777)
        LOGGER.info(f"🔌 PTZ 服务器监听: {self.socket_path}")

        # 启动工作线程
        self.worker.start()

        # 主线程：处理客户端连接
        self._running.set()
        self.server_socket.settimeout(1.0)
        LOGGER.info("🚀 PTZ 服务器已启动，等待连接...")

        try:
            while self._running.is_set():
                try:
                    conn, addr = self.server_socket.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break

                # 每个客户端连接在独立线程中处理
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(conn,),
                    daemon=True,
                    name=f"client-{time.time()}"
                )
                client_thread.start()

        except KeyboardInterrupt:
            LOGGER.info("收到 SIGINT，正在关闭...")
        finally:
            self.shutdown()

    def shutdown(self):
        """关闭服务器"""
        LOGGER.info("🛑 正在关闭 PTZ 服务器...")
        self._running.clear()
        self._stop_signaled.set()

        # 停止工作线程
        self.worker.stop()

        # 清空命令队列
        while not self.cmd_queue.empty():
            try:
                self.cmd_queue.get_nowait()
                self.cmd_queue.task_done()
            except Empty:
                break

        # 关闭 socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass

        # 清理 socket 文件
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass

        # 关闭云台
        try:
            self.ptz.close()
        except Exception:
            pass

        LOGGER.info("✅ PTZ 服务器已关闭")

    def _handle_client(self, conn):
        """处理单个客户端连接"""
        conn.settimeout(30.0)
        buffer = b""
        try:
            while self._running.is_set() and not self._stop_signaled.is_set():
                try:
                    data = conn.recv(4096)
                except socket.timeout:
                    continue
                except (ConnectionResetError, BrokenPipeError, OSError):
                    break

                if not data:
                    break

                buffer += data

                # 尝试解析完整的 JSON 请求（支持批量发送）
                while buffer:
                    # 按 \n 拆分成多个命令
                    if b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                    else:
                        # 没有换行符时尝试解析整个 buffer（之后不再进内层循环）
                        line = buffer.strip()
                        buffer = b""

                    if not line:
                        continue

                    try:
                        request = json.loads(line.decode("utf-8"))
                    except (json.JSONDecodeError, ValueError):
                        LOGGER.warning(f"JSON 解析失败: {line[:100]}")
                        continue

                    response = self._process_request(request)
                    try:
                        conn.sendall((json.dumps(response) + "\n").encode("utf-8"))
                    except (BrokenPipeError, ConnectionResetError):
                        return

        except Exception as e:
            LOGGER.warning(f"客户端处理异常: {e}")
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _process_request(self, request):
        """处理单个请求"""
        cmd = request.get("cmd", "")
        response_queue = Queue()
        pt_state = self.ptz.get_state()

        # ---- 即时命令（不经过队列） ----
        if cmd == "get_position":
            return {"ok": True, **pt_state}

        elif cmd == "get_status":
            return {
                "ok": True,
                **pt_state,
                "queue_depth": self.cmd_queue.qsize(),
                "worker_alive": self.worker.is_alive(),
            }

        elif cmd == "clear":
            # 跳过当前命令 + 清空队列
            self.worker.skip_current()
            while not self.cmd_queue.empty():
                try:
                    self.cmd_queue.get_nowait()
                    self.cmd_queue.task_done()
                except Empty:
                    break
            self.ptz.stop()
            return {"ok": True, "msg": "队列已清空", **self.ptz.get_state()}

        elif cmd == "stop":
            self.worker.skip_current()
            self.ptz.stop()
            return {"ok": True, "msg": "已停止", **self.ptz.get_state()}

        elif cmd == "shutdown":
            # 在独立线程中关闭，以免阻塞响应
            threading.Thread(target=self.shutdown, daemon=True).start()
            return {"ok": True, "msg": "服务器正在关闭"}

        elif cmd == "set_limits":
            self.ptz.set_limits(
                pan_min=request.get("pan_min"),
                pan_max=request.get("pan_max"),
                tilt_min=request.get("tilt_min"),
                tilt_max=request.get("tilt_max"),
            )
            return {"ok": True, **self.ptz.get_state()}

        # ---- 运动命令（入队串行执行） ----
        elif cmd in ("move_to", "move_pan", "move_tilt", "home", "reset_pan", "reset_tilt"):
            cmd_obj = PtzCommand(cmd, request, response_queue)
            self.cmd_queue.put(cmd_obj)

            # 阻塞等待执行结果
            try:
                result = response_queue.get(timeout=180.0)
            except Empty:
                return {"ok": False, "error": "命令执行超时 (180s)", **pt_state}

            return result

        else:
            return {"ok": False, "error": f"未知命令: {cmd}", **pt_state}


# ============================================================
# 启动/停止管理
# ============================================================

def write_pid(pid):
    with open(PID_FILE, "w") as f:
        f.write(str(pid))


def read_pid():
    try:
        with open(PID_FILE, "r") as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return None


def is_running():
    pid = read_pid()
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def cmd_start(args):
    if is_running():
        print("PTZ 服务器已在运行中")
        return

    pid = os.fork()
    if pid > 0:
        # 父进程
        write_pid(pid)
        print(f"PTZ 服务器已启动 (PID={pid})")
        print(f"  Socket: {SOCKET_PATH}")
        print(f"  Port: {args.port} @ {args.baudrate} baud")
        print(f"  Initial: pan={args.initial_pan}°, tilt={args.initial_tilt}°")
        sys.exit(0)

    # 子进程：启动服务器
    os.setsid()
    # 重定向标准 I/O（后台守护进程）
    with open("/dev/null", "r") as f:
        os.dup2(f.fileno(), sys.stdin.fileno())
    with open("/tmp/ptz_server.log", "a") as f:
        os.dup2(f.fileno(), sys.stdout.fileno())
        os.dup2(f.fileno(), sys.stderr.fileno())

    server = PtzServer(
        port=args.port,
        baudrate=args.baudrate,
        initial_pan=args.initial_pan,
        initial_tilt=args.initial_tilt,
    )
    server.start()


def cmd_stop():
    if not is_running():
        print("PTZ 服务器未运行")
        return

    client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client_socket.connect(SOCKET_PATH)
        client_socket.sendall(b'{"cmd":"shutdown"}\n')
        client_socket.settimeout(3.0)
        try:
            data = client_socket.recv(4096)
            print(f"响应: {data.decode()}")
        except socket.timeout:
            pass
    except (ConnectionRefusedError, FileNotFoundError) as e:
        print(f"无法连接 PTZ 服务器: {e}")
        # 强制 kill
        pid = read_pid()
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"已通过 SIGTERM 停止 (PID={pid})")
            except ProcessLookupError:
                print("进程不存在")
    finally:
        client_socket.close()

    # 清理 pid 文件
    if os.path.exists(PID_FILE):
        os.unlink(PID_FILE)
    if os.path.exists(SOCKET_PATH):
        try:
            os.unlink(SOCKET_PATH)
        except OSError:
            pass
    print("PTZ 服务器已停止")


def cmd_restart(args):
    cmd_stop()
    time.sleep(1)
    cmd_start(args)


def cmd_status():
    pid = read_pid()
    running = is_running()
    print(f"PTZ 服务器: {'🟢 运行中' if running else '🔴 已停止'}")
    if running:
        print(f"  PID: {pid}")
        print(f"  Socket: {SOCKET_PATH}")

        # 尝试查询状态
        try:
            import ptz_client
            client = ptz_client.PTZClient()
            if client.connect():
                status = client.get_status()
                if status.get("ok"):
                    print(f"  当前位置: pan={status.get('pan','?')}°, tilt={status.get('tilt','?')}°")
                    print(f"  范围: pan [{status.get('pan_min','?')}, {status.get('pan_max','?')}], "
                          f"tilt [{status.get('tilt_min','?')}, {status.get('tilt_max','?')}]")
                    print(f"  队列深度: {status.get('queue_depth', '?')}")
                    print(f"  工作线程: {'🟢' if status.get('worker_alive') else '🔴'}")
                client.close()
        except Exception:
            print("  （无法连接查询）")


# ============================================================
# CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="PTZ 控制服务器")
    subparsers = parser.add_subparsers(dest="action", required=True)

    start_parser = subparsers.add_parser("start", help="启动 PTZ 服务器")
    start_parser.add_argument("--port", default=DEFAULT_PORT)
    start_parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE)
    start_parser.add_argument("--initial-pan", type=float, default=DEFAULT_INITIAL_PAN)
    start_parser.add_argument("--initial-tilt", type=float, default=DEFAULT_INITIAL_TILT)

    subparsers.add_parser("stop", help="停止 PTZ 服务器")
    subparsers.add_parser("restart", help="重启 PTZ 服务器")
    subparsers.add_parser("status", help="查看 PTZ 服务器状态")

    args = parser.parse_args()

    if args.action == "start":
        cmd_start(args)
    elif args.action == "stop":
        cmd_stop()
    elif args.action == "restart":
        cmd_restart(args)
    elif args.action == "status":
        cmd_status()


if __name__ == "__main__":
    main()
