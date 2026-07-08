# Home Assistant — 机器人工作空间

本项目为 **2026 年全国大学生嵌入式芯片与系统设计竞赛（应用赛道）** 参赛作品，由 **ASC-EAI 团队** 独立完成。

基于 **RDK S100P** 边缘计算平台，汇聚了一套**双臂机器人系统**的全部软件组件，涵盖视觉-语言-动作（VLA）推理、面部表情识别、语音交互、工作区域安全检测等能力。

---

## 目录结构

| 目录 | 说明 |
|---|---|
| `RoboOrchard/` | 核心机器人框架：ROS2 控制、CAN 总线、piper 机械臂 SDK、RealSense 相机驱动、VLA 部署 |
| `HoloBrain_ws/` | VLA 模型文件（HBM 格式）及推理服务端脚本：物体抓取、摇瓶子、整理餐具 |
| `Face_Emotion/` | 实时面部表情识别（YOLOv8 BPU 人脸检测 + MobileFaceNet 情绪分类） |
| `Voice/` | 语音模块串口桥接、TTS FIFO 监听、命令转发到 OpenClaw 会话 |
| `person_distance/` | 人员检测 + PTZ 云台跟踪 + 安全告警（YOLO + RealSense 深度相机） |
| `environtment.yml` | Conda 环境定义（`holo` 环境） |
| `requirements.txt` | pip 依赖清单 |

## 快速开始

> 以下命令均在项目根目录 `~/Data/home_assistant/` 下执行。激活环境后，根据需求选择对应模块启动。

### 1. 激活环境

```bash
conda activate holo
```

### 2. 表情识别

```bash
# 1 秒轮询检测人脸表情，检测到 sad 自动触发送飞书通知 + TTS 播报 + 摇瓶子工作流
python3 Face_Emotion/emotion.py --no-preview
```

### 3. 语音模块桥接（可选）

```bash
# 监听语音模块串口，识别到的命令转发到 OpenClaw 会话
python3 Voice/oc_voice_session.py monitor --execute --session-key <your-session-key>
```

### 4. 人员检测 + 安全告警（可选）

```bash
# 启动 PTZ 云台 + YOLO 人员检测，闯入区域自动飞书告警并停止推理
cd person_distance && python3 computer_security.py
```

### 5. VLA 推理服务

各 VLA 任务有独立的推理服务端：

```bash
# 物体抓取
python3 HoloBrain_ws/server_grasp_anything.py

# 摇瓶子
python3 HoloBrain_ws/server_shakebottle.py

# 整理餐具
python3 HoloBrain_ws/server_tableware.py
```

### 6. 机械臂控制（ROS2）

```bash
# 需要先激活 ROS2 环境并启动机械臂节点
source RoboOrchard/ros2_package/install/setup.bash
# 双臂回零
ros2 service call /robot/left/reset_ctrl std_srvs/srv/Trigger "{}"
ros2 service call /robot/right/reset_ctrl std_srvs/srv/Trigger "{}"
```

## 系统架构

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│  语音模块    │────→│  语音桥接     │────→│ OpenClaw 会话    │
│  (串口)      │     │  oc_voice    │     │ (小瓜智能体)      │
└─────────────┘     └──────────────┘     └────────┬─────────┘
                                                  │
                    ┌─────────────────────────────┼─────────────────────────────┐
                    │                             │                             │
          ┌─────────▼─────────┐       ┌───────────▼──────────┐      ┌──────────▼──────────┐
          │    Face_Emotion   │       │    person_distance   │      │     RoboOrchard     │
          │  YOLOv8 人脸检测  │       │  YOLO 人员检测       │      │  ROS2 + CAN 总线     │
          │  MobileFaceNet    │       │  RealSense D435I     │      │  Piper 双臂          │
          │  情绪分类         │       │  PTZ 云台跟踪        │      │  HoloBrain VLA       │
          └─────────┬─────────┘       └───────────┬──────────┘      └──────────┬──────────┘
                    │                             │                           │
              RS D435I 相机                  PTZ 云台相机                 双臂机械臂
```

## 依赖环境

| 依赖 | 版本/说明 |
|---|---|
| ROS2 Humble | 机器人操作系统 |
| librealsense2 | RealSense 相机 SDK |
| ONNX Runtime / hbm_runtime | 模型推理引擎 |
| Conda（`holo` 环境） | Python 依赖管理 |
| OpenClaw Gateway | 会话层命令路由 |

## 许可证

参见项目根目录的 `LICENSE` 文件（MIT License）。
