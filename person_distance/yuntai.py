import serial
import time
import warnings

class PanTiltController:
    """
    云台控制器，支持任意位置控制（带速度、范围限制）。
    提供角度校准功能，修正内部记录与实际物理位置的偏差。
    """

    # 预定义指令
    CMD_UP    = bytes.fromhex('FF 01 00 10 00 20 31')
    CMD_DOWN  = bytes.fromhex('FF 01 00 08 00 20 29')
    CMD_LEFT  = bytes.fromhex('FF 01 00 04 20 00 25')
    CMD_RIGHT = bytes.fromhex('FF 01 00 02 20 00 23')
    CMD_STOP  = bytes.fromhex('FF 01 00 00 00 00 01')

    # 速度（度/秒）
    PAN_SPEED   = 5.0
    TILT_SPEED  = 3.0   # ≈ 3.333

    # 角度范围
    PAN_MIN, PAN_MAX   = 0.0, 180.0
    TILT_MIN, TILT_MAX = 0.0, 90.0

    def __init__(self, port='/dev/ptz', baudrate=9600, timeout=1,
                 initial_pan=0.0, initial_tilt=0.0):
        """
        :param initial_pan:  认为当前物理 pan 角度为多少（默认 0）
        :param initial_tilt: 认为当前物理 tilt 角度为多少（默认 0）
        """
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        # 将内部记录设置为用户指定的初始值（裁剪到合法范围）
        self.pan = self._clamp(initial_pan, self.PAN_MIN, self.PAN_MAX)
        self.tilt = self._clamp(initial_tilt, self.TILT_MIN, self.TILT_MAX)
        self.__init_motion_state()

    @staticmethod
    def _clamp(value, min_val, max_val):
        if value < min_val:
            warnings.warn(f"值 {value} 低于最小值 {min_val}，已裁剪至 {min_val}")
            return min_val
        if value > max_val:
            warnings.warn(f"值 {value} 超过最大值 {max_val}，已裁剪至 {max_val}")
            return max_val
        return value

    def _send(self, cmd_bytes):
        self.ser.write(cmd_bytes)
        time.sleep(0.05)

    def _move_relative(self, direction_cmd, angle, speed):
        """向指定方向转动 angle 度（angle>0），然后停止"""
        if angle <= 0:
            return
        self._send(direction_cmd)
        time.sleep(angle / speed)
        self._send(self.CMD_STOP)

    # ---------- 核心位置控制 ----------
    def set_pan(self, target_pan):
        """设定水平目标角度（阻塞版，等待运动完成）"""
        target_pan = self._clamp(target_pan, self.PAN_MIN, self.PAN_MAX)
        diff = target_pan - self.pan
        if diff > 0:
            self._move_relative(self.CMD_RIGHT, diff, self.PAN_SPEED)
        elif diff < 0:
            self._move_relative(self.CMD_LEFT, -diff, self.PAN_SPEED)
        self.pan = target_pan

    def set_tilt(self, target_tilt):
        """设定俯仰目标角度（阻塞版）"""
        target_tilt = self._clamp(target_tilt, self.TILT_MIN, self.TILT_MAX)
        diff = target_tilt - self.tilt
        if diff > 0:
            self._move_relative(self.CMD_UP, diff, self.TILT_SPEED)
        elif diff < 0:
            self._move_relative(self.CMD_DOWN, -diff, self.TILT_SPEED)
        self.tilt = target_tilt

    # ---------- 非阻塞位置控制（用于周期性调用的主循环） ----------
    def __init_motion_state(self):
        """初始化非阻塞运动状态（pan/tilt 独立跟踪）"""
        self._pan_target = None
        self._pan_start = None
        self._pan_dur = None
        self._pan_dir = None
        self._pan_just_completed = False
        self._tilt_target = None
        self._tilt_start = None
        self._tilt_dur = None
        self._tilt_dir = None
        self._tilt_just_completed = False

    def start_move_pan(self, target_pan):
        """
        非阻塞开始水平转动。
        发送方向指令后立即返回，由外部循环周期性调用 update_motion() 停止。
        """
        target_pan = self._clamp(target_pan, self.PAN_MIN, self.PAN_MAX)
        diff = target_pan - self.pan
        if abs(diff) < 1.0:
            self._pan_target = None
            return
        direction = self.CMD_RIGHT if diff > 0 else self.CMD_LEFT
        dur = abs(diff) / self.PAN_SPEED
        self._send(direction)
        self._pan_target = target_pan
        self._pan_start = time.time()
        self._pan_dur = dur
        self._pan_dir = direction

    def start_move_tilt(self, target_tilt):
        """非阻塞开始俯仰转动"""
        target_tilt = self._clamp(target_tilt, self.TILT_MIN, self.TILT_MAX)
        diff = target_tilt - self.tilt
        if abs(diff) < 1.0:
            self._tilt_target = None
            return
        direction = self.CMD_UP if diff > 0 else self.CMD_DOWN
        dur = abs(diff) / self.TILT_SPEED
        self._send(direction)
        self._tilt_target = target_tilt
        self._tilt_start = time.time()
        self._tilt_dur = dur
        self._tilt_dir = direction

    def update_motion(self):
        """
        检查非阻塞运动完成、更新内部角度。
        外部循环每帧调用。
        返回 (pan_just_done, tilt_just_done) — 仅在该帧刚完成时为 True。
        """
        pan_just = False
        tilt_just = False
        self._pan_just_completed = False
        self._tilt_just_completed = False
        now = time.time()

        if self._pan_start is not None:
            if now - self._pan_start >= self._pan_dur:
                self._send(self.CMD_STOP)
                if self._pan_target is not None:
                    self.pan = self._pan_target
                    self._pan_target = None
                self._pan_start = None
                self._pan_dur = None
                self._pan_just_completed = True
                pan_just = True
        if self._tilt_start is not None:
            if now - self._tilt_start >= self._tilt_dur:
                self._send(self.CMD_STOP)
                if self._tilt_target is not None:
                    self.tilt = self._tilt_target
                    self._tilt_target = None
                self._tilt_start = None
                self._tilt_dur = None
                self._tilt_just_completed = True
                tilt_just = True
        return pan_just, tilt_just

    def is_moving(self):
        return (self._pan_start is not None) or (self._tilt_start is not None)

    # ---------- 角度校准（关键新增） ----------
    def set_current_pan(self, actual_pan):
        """
        校准水平角度：告诉控制器当前物理 pan 的实际值（不移动云台）。
        之后的所有 pan 运动将基于此值计算差值。
        """
        self.pan = self._clamp(actual_pan, self.PAN_MIN, self.PAN_MAX)

    def set_current_tilt(self, actual_tilt):
        """
        校准俯仰角度：告诉控制器当前物理 tilt 的实际值（不移动云台）。
        """
        self.tilt = self._clamp(actual_tilt, self.TILT_MIN, self.TILT_MAX)

    def go_home(self):
        """一键回零（pan=0, tilt=0），前提是内部记录已正确校准"""
        self.set_pan(0)
        self.set_tilt(0)

    # ---------- 便捷步进（范围保护） ----------
    def move_up(self):
        self.set_tilt(self.tilt + 90)

    def move_down(self):
        self.set_tilt(self.tilt - 90)

    def move_left(self):
        self.set_pan(self.pan - 90)

    def move_right(self):
        self.set_pan(self.pan + 90)

    def stop(self):
        self._send(self.CMD_STOP)

    def close(self):
        self.ser.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ========== 使用示例（强调校准） ==========
if __name__ == '__main__':
    with PanTiltController('/dev/ptz', 9600) as pt:
        # ------------------------------
        # 场景：云台当前物理位置未知，我们需要先校准
        # 假设我们手动将云台转到了 pan=30°, tilt=20°（例如通过手动拧或之前运动）
        # 那么在代码中，我们应该告诉控制器当前真实角度：
        pt.set_current_pan(60)   # 告诉控制器：当前 pan 实际为 30°
        pt.set_current_tilt(0)  # 当前 tilt 实际为 20°
        print(f"校准后，内部记录: pan={pt.pan}°, tilt={pt.tilt}°")

        # 现在我们可以自由设定目标，计算差值是准确的
        pt.set_pan(0)            # 从 30° 移动到 0°，需要向左转 30°
        pt.set_tilt(0)           # 从 20° 移动到 0°，需要向下转 20°
        print(f"回零后: pan={pt.pan}°, tilt={pt.tilt}°")

        # 也可以直接调用 go_home()（但内部必须已校准）
        # pt.go_home()

        # 再设定其他角度
        # pt.set_pan(90)
        # pt.set_tilt(45)
        print(f"移动后: pan={pt.pan}°, tilt={pt.tilt}°")
        pt.stop()
