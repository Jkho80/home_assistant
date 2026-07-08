#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ptz_client.py — PTZ 控制客户端
===============================
PTZ 服务器的客户端库。提供便捷的 Python API。

用法：
    from ptz_client import PTZClient

    client = PTZClient()
    if client.connect():
        # 非阻塞命令（入队后立即返回执行结果）
        result = client.move_pan(90.0)       # 水平移动
        result = client.move_tilt(45.0)      # 俯仰移动
        result = client.home()               # 回到初始位置

        # 阻塞命令（等待执行完成）
        result = client.move_to(90.0, 45.0)  # 移动到指定位置

        # 查询
        pos = client.get_position()          # 获取当前位置
        status = client.get_status()         # 获取状态

        # 控制
        result = client.stop()               # 停止运动
        result = client.clear()              # 清空队列
        result = client.set_limits(pan_min=0, pan_max=180)
        result = client.reset_pan(90.0)      # 校准

        client.close()

CLI 用法：
    python3 ptz_client.py position          # 查询当前位置
    python3 ptz_client.py move_to 90 45     # 移动
    python3 ptz_client.py move_pan 90       # 水平移动
    python3 ptz_client.py stop              # 停止
    python3 ptz_client.py status            # 详细状态
"""

import json
import os
import socket
import sys
import time

SOCKET_PATH = "/tmp/ptz_server.sock"
DEFAULT_TIMEOUT = 30.0


class PTZClient:
    """PTZ 服务器客户端"""

    def __init__(self, socket_path=SOCKET_PATH, timeout=DEFAULT_TIMEOUT):
        self.socket_path = socket_path
        self.timeout = timeout
        self.sock = None

    def connect(self):
        """连接到 PTZ 服务器"""
        if self.sock:
            return True
        try:
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.settimeout(self.timeout)
            self.sock.connect(self.socket_path)
            return True
        except (ConnectionRefusedError, FileNotFoundError, OSError) as e:
            self.sock = None
            return False

    def close(self):
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def _send_command(self, cmd_dict, timeout=None):
        """发送命令并接收响应"""
        if not self.sock:
            raise ConnectionError("未连接到 PTZ 服务器")

        # 临时调整 socket 超时
        orig_timeout = self.sock.gettimeout()
        effective_timeout = timeout or self.timeout
        self.sock.settimeout(min(10.0, effective_timeout))

        try:
            self.sock.sendall((json.dumps(cmd_dict) + "\n").encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError):
            self.close()
            raise ConnectionError("PTZ 服务器连接断开")

        # 接收响应
        buffer = b""
        deadline = time.time() + effective_timeout
        while time.time() < deadline:
            try:
                data = self.sock.recv(4096)
            except socket.timeout:
                continue
            except (ConnectionResetError, BrokenPipeError, OSError):
                self.close()
                raise ConnectionError("PTZ 服务器连接断开")

            if not data:
                self.close()
                raise ConnectionError("PTZ 服务器连接关闭")

            buffer += data
            try:
                response = json.loads(buffer.decode("utf-8").strip())
                return response
            except json.JSONDecodeError:
                continue

        self.sock.settimeout(orig_timeout)
        raise TimeoutError("PTZ 服务器响应超时")

    # ========== 便利方法 ==========    # ========== 便利方法 ==========    # ========== 便利方法 ==========

    def move_pan(self, pan, timeout=120.0):
        """水平移动（非阻塞：入队等待执行，返回结果）"""
        return self._send_command({"cmd": "move_pan", "pan": pan}, timeout)

    def move_tilt(self, tilt, timeout=120.0):
        """俯仰移动（非阻塞：入队等待执行，返回结果）"""
        return self._send_command({"cmd": "move_tilt", "tilt": tilt}, timeout)

    def move_to(self, pan, tilt, timeout=180.0):
        """移动到指定位置（阻塞：等待全部运动完成）"""
        return self._send_command({"cmd": "move_to", "pan": pan, "tilt": tilt}, timeout)

    def home(self, timeout=120.0):
        """回到初始位置"""
        return self._send_command({"cmd": "home"}, timeout)

    def get_position(self):
        """查询当前位置（立即返回）"""
        return self._send_command({"cmd": "get_position"}, timeout=5.0)

    def get_status(self):
        """查询详细状态（立即返回）"""
        return self._send_command({"cmd": "get_status"}, timeout=5.0)

    def stop(self):
        """停止当前运动（立即执行）"""
        return self._send_command({"cmd": "stop"}, timeout=5.0)

    def clear(self):
        """清空队列并停止（立即执行）"""
        return self._send_command({"cmd": "clear"}, timeout=5.0)

    def reset_pan(self, pan):
        """校准当前物理 pan 角度（不移动）"""
        return self._send_command({"cmd": "reset_pan", "pan": pan}, timeout=5.0)

    def reset_tilt(self, tilt):
        """校准当前物理 tilt 角度（不移动）"""
        return self._send_command({"cmd": "reset_tilt", "tilt": tilt}, timeout=5.0)

    def set_limits(self, pan_min=None, pan_max=None, tilt_min=None, tilt_max=None):
        """设置角度限制"""
        cmd = {"cmd": "set_limits"}
        if pan_min is not None: cmd["pan_min"] = pan_min
        if pan_max is not None: cmd["pan_max"] = pan_max
        if tilt_min is not None: cmd["tilt_min"] = tilt_min
        if tilt_max is not None: cmd["tilt_max"] = tilt_max
        return self._send_command(cmd, timeout=5.0)


# ========== CLI ==========

def cli_help():
    print("用法: python3 ptz_client.py <command> [args...]")
    print()
    print("命令:")
    print("  position               查询当前位置")
    print("  status                 查询详细状态")
    print("  move_to <pan> <tilt>   移动到指定位置")
    print("  move_pan <pan>         水平移动")
    print("  move_tilt <tilt>       俯仰移动")
    print("  home                   回到初始位置")
    print("  stop                   停止运动")
    print("  clear                  清空队列并停止")
    print("  reset_pan <pan>        校准 pan")
    print("  reset_tilt <tilt>      校准 tilt")
    print("  set_limits [pan_min] [pan_max] [tilt_min] [tilt_max]")
    sys.exit(0)


def cli_main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        cli_help()

    cmd = sys.argv[1]
    args = sys.argv[2:]

    client = PTZClient()
    if not client.connect():
        print("❌ 无法连接到 PTZ 服务器（/tmp/ptz_server.sock）")
        print("   请先启动: python3 ptz_server.py start")
        sys.exit(1)

    try:
        if cmd == "position":
            result = client.get_position()
        elif cmd == "status":
            result = client.get_status()
        elif cmd == "move_to":
            if len(args) < 2:
                print("需要 pan 和 tilt 两个参数")
                sys.exit(1)
            result = client.move_to(float(args[0]), float(args[1]))
        elif cmd == "move_pan":
            if len(args) < 1:
                print("需要 pan 参数")
                sys.exit(1)
            result = client.move_pan(float(args[0]))
        elif cmd == "move_tilt":
            if len(args) < 1:
                print("需要 tilt 参数")
                sys.exit(1)
            result = client.move_tilt(float(args[0]))
        elif cmd == "home":
            result = client.home()
        elif cmd == "stop":
            result = client.stop()
        elif cmd == "clear":
            result = client.clear()
        elif cmd == "reset_pan":
            if len(args) < 1:
                print("需要 pan 参数")
                sys.exit(1)
            result = client.reset_pan(float(args[0]))
        elif cmd == "reset_tilt":
            if len(args) < 1:
                print("需要 tilt 参数")
                sys.exit(1)
            result = client.reset_tilt(float(args[0]))
        elif cmd == "set_limits":
            kwargs = {}
            if len(args) >= 1 and args[0] != "_": kwargs["pan_min"] = float(args[0])
            if len(args) >= 2 and args[1] != "_": kwargs["pan_max"] = float(args[1])
            if len(args) >= 3 and args[2] != "_": kwargs["tilt_min"] = float(args[2])
            if len(args) >= 4 and args[3] != "_": kwargs["tilt_max"] = float(args[3])
            result = client.set_limits(**kwargs)
        else:
            print(f"未知命令: {cmd}")
            cli_help()
    except Exception as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)
    finally:
        client.close()

    def _fmt(val):
        if val is None:
            return "?"
        return val

    if result.get("ok"):
        line = f"✅ OK"
        if "pan" in result:
            line += f" | pan={_fmt(result['pan'])}°"
        if "tilt" in result:
            line += f" | tilt={_fmt(result['tilt'])}°"
        if "queue_depth" in result:
            line += f" | 队列={result['queue_depth']}"
        if "msg" in result:
            line += f" | {result['msg']}"
        if "error" in result:
            line += f" | ⚠️ {result['error']}"
        print(line)

        if cmd in ("status",):
            print(f"  范围: pan [{_fmt(result.get('pan_min'))}, {_fmt(result.get('pan_max'))}], "
                  f"tilt [{_fmt(result.get('tilt_min'))}, {_fmt(result.get('tilt_max'))}]")
            print(f"  工作线程: {'🟢 正常' if result.get('worker_alive') else '🔴 异常'}")
    else:
        print(f"❌ 失败: {result.get('error', '未知错误')}")


if __name__ == "__main__":
    cli_main()
