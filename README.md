<<<<<<< HEAD
# home_assistant
2026年全国大学生 嵌入式芯片与系统设计竞赛 参赛作品 《智护家园—面向家庭安全陪护的双臂智能机器人》
=======
# Home Assistant — Robotic Workspace

This project is an entry for the **2026 National Undergraduate Embedded Chip & System Design Competition (Application Track)**, independently developed by the **ASC-EAI Team**.

Built on the **RDK S100P** edge computing platform, it aggregates all software components of a **dual-arm robotic system**, including Vision-Language-Action (VLA) inference, facial emotion recognition, voice interaction, and workspace safety detection.

---

## Directory Structure

| Directory | Description |
|---|---|
| `RoboOrchard/` | Core robot framework: ROS2 control, CAN bus, piper arm SDK, RealSense camera driver, VLA deployment |
| `HoloBrain_ws/` | VLA model artifacts (HBM format) and inference server scripts: grasp anything, shake bottle, organize tableware |
| `Face_Emotion/` | Real-time facial emotion detection (YOLOv8 BPU face detection + MobileFaceNet emotion classification) |
| `Voice/` | Voice module serial bridge, TTS FIFO listener, command forwarding to OpenClaw session |
| `person_distance/` | Person detection + PTZ tracking + safety alerts (YOLO + RealSense depth camera) |
| `environtment.yml` | Conda environment definition (`holo` env) |
| `requirements.txt` | pip dependency list |

## Quick Start

> Run all commands from the project root `~/Data/home_assistant/`. Activate the Conda environment first, then launch the desired modules.

### 1. Activate Environment

```bash
conda activate holo
```

### 2. Facial Emotion Detection

```bash
# 1-second polling loop for face emotion detection.
# Triggers Feishu notification + TTS + shake-bottle workflow on "sad" detection.
python3 Face_Emotion/emotion.py --no-preview
```

### 3. Voice Module Bridge *(optional)*

```bash
# Listens to the voice module serial port and forwards recognized commands
# to the OpenClaw session.
python3 Voice/oc_voice_session.py monitor --execute --session-key <your-session-key>
```

### 4. Person Detection & Safety Alerts *(optional)*

```bash
# Starts PTZ camera + YOLO person detection.
# Sends Feishu alerts and disables inference when someone enters the danger zone.
cd person_distance && python3 computer_security.py
```

### 5. VLA Inference Servers

Each VLA task has a dedicated inference server:

```bash
# Grasp anything
python3 HoloBrain_ws/server_grasp_anything.py

# Shake bottle
python3 HoloBrain_ws/server_shakebottle.py

# Organize tableware
python3 HoloBrain_ws/server_tableware.py
```

### 6. Robot Arm Control (ROS2)

```bash
# Activate ROS2 environment and start the arm nodes first
source RoboOrchard/ros2_package/install/setup.bash

# Reset both arms to home position
ros2 service call /robot/left/reset_ctrl std_srvs/srv/Trigger "{}"
ros2 service call /robot/right/reset_ctrl std_srvs/srv/Trigger "{}"
```

## System Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│   Voice     │     │  Voice       │     │  OpenClaw        │
│   Module    │────→│  Bridge      │────→│  Session (小瓜)   │
│  (Serial)   │     │  oc_voice    │     │                  │
└─────────────┘     └──────────────┘     └────────┬─────────┘
                                                   │
                    ┌──────────────────────────────┼──────────────────────────────┐
                    │                              │                              │
          ┌─────────▼──────────┐       ┌───────────▼──────────┐      ┌───────────▼──────────┐
          │   Face_Emotion     │       │   person_distance    │      │     RoboOrchard      │
          │  YOLOv8 Face Det.  │       │  YOLO Person Det.    │      │  ROS2 + CAN Bus      │
          │  MobileFaceNet     │       │  RealSense D435I     │      │  Piper Dual Arms     │
          │  Emotion Classif.  │       │  PTZ Tracking        │      │  HoloBrain VLA       │
          └─────────┬──────────┘       └───────────┬──────────┘      └───────────┬──────────┘
                    │                              │                            │
              RS D435I Camera              PTZ Camera                     Dual Arms
```

## Dependencies

| Dependency | Version / Notes |
|---|---|
| ROS2 Humble | Robot operating system |
| librealsense2 | RealSense camera SDK |
| ONNX Runtime / hbm_runtime | Model inference engine |
| Conda (`holo` env) | Python dependency management |
| OpenClaw Gateway | Session-level command routing |

## License

See the `LICENSE` file in the project root (MIT License).
>>>>>>> master
