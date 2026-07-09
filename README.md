> A home desktop dual-arm robot that can be controlled through Feishu or voice, observe the tabletop, tidy up objects, shake drinks, and also knows when it should not move.

*README_en.md of Chinese Version please refer in README.md.*

# Home Assistant — OpenClaw Home Desktop Robot Workspace

This project is an entry for the **2026 National College Student Embedded Chip and System Design Competition — Application Track**, independently developed by the **ASC-EAI Team**.

The project targets home desktop service, remote observation, lightweight companionship, and safe human-robot interaction scenarios. Built on the **RDK S100P** edge computing platform, it provides a complete software workspace for a dual-arm robotic system. The system integrates Vision-Language-Action (VLA) inference, desktop object detection, facial expression recognition, voice interaction, Feishu remote interaction, human proximity detection, and robotic arm state monitoring.

Users can issue natural-language commands to the robot through **local voice input** or **Feishu text messages**, such as remotely checking the tabletop, tidying up objects, shaking a drink, or querying the robot status. The OpenClaw Agent is responsible for understanding user intent and scheduling skills, while the RDK S100P handles edge-side model inference, camera access, robotic arm service calls, and safety monitoring. Together, they form a robotic closed loop based on “low-frequency cloud/PC-side agent coordination + high-frequency edge-side model execution”.

---

## Project Positioning

This project is not a single robotic arm control program. Instead, it is an integrated robotic system workspace for home desktop scenarios, focusing on the following capabilities:

1. **Natural Language Interaction**
   - Supports Feishu text commands.
   - Supports local voice module input.
   - Supports Web / OpenClaw conversation entry.
   - Users do not need to directly operate low-level robotic arm control interfaces.

2. **Remote Scene Understanding**
   - Observes the tabletop through a global desktop-view camera and a third-person PTZ camera.
   - Supports remote checking of tabletop status, human presence, and target object status.
   - Supports returning text status, images, or keyframe results through Feishu.

3. **Edge-side VLA Manipulation**
   - Uses HoloBrain / VLA models to generate robot actions for desktop tasks.
   - Supports low-risk operations such as desktop tidying, object pick-and-place, and drink shaking.
   - The model has been exported to ONNX, compiled into HBM, and deployed on the RDK S100P.

4. **Lightweight Companionship Interaction**
   - Uses face detection and expression recognition to perceive user state.
   - When a predefined low-mood expression is detected, the system can trigger voice comfort and a drink-shaking task.
   - Emotion recognition is only used for lightweight companionship feedback, not for medical or psychological diagnosis.

5. **Safe Execution and Abnormal State Alerts**
   - Supports rejection of dangerous instructions.
   - Supports refusing blind execution when the requested target does not exist.
   - Supports stopping inference and action dispatch when a person approaches the robotic arm workspace.
   - Supports reporting robotic arm abnormal states and notifying the user through Feishu.

---

## Directory Structure

| Directory / File | Description |
|---|---|
| `RoboOrchard/` | Core robot framework, including ROS2 control, CAN bus, Piper robotic arm SDK, RealSense camera drivers, and VLA deployment-related code |
| `HoloBrain_ws/` | VLA model files and inference server scripts, including object grasping, bottle shaking, and tableware organization tasks |
| `Face_Emotion/` | Real-time facial expression recognition module, including YOLOv8 BPU face detection and MobileFaceNet emotion classification |
| `Voice/` | Voice module serial bridge, TTS FIFO listener, and voice command forwarding to OpenClaw sessions |
| `person_distance/` | Human detection, RealSense depth estimation, PTZ tracking, and safety alert module |
| `environtment.yml` | Conda environment definition, with the default environment name `holo` |
| `requirements.txt` | Python pip dependency list |
| `README.md` | Project documentation |

---

## System Architecture

The system uses a five-layer architecture:

```text
User Interaction Layer
  ├── Local Voice
  ├── Feishu Text Messages
  └── Web / OpenClaw Session

Agent Task Layer
  └── OpenClaw Agent
      ├── Natural Language Understanding
      ├── Semantic Safety Judgment
      ├── Skill Selection and Task Scheduling
      └── Status Feedback

Restricted Skill Library Layer
  ├── Scene Understanding
  ├── Manipulation Execution
  ├── Companionship Interaction
  └── Safety Recovery

RDK S100P Edge Execution Layer
  ├── YOLO Object Detection
  ├── HoloBrain VLA Inference
  ├── Face / Expression Recognition
  ├── RGB-D Safety Detection
  ├── PTZ Control
  └── Robotic Arm Service Calls

Robot Hardware Layer
  ├── Piper Dual-arm Robot
  ├── RealSense D435 / D435i RGB-D Cameras
  ├── Third-person PTZ Camera
  └── CI1302 Voice Module
```

Simplified architecture diagram:

```text
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│ Voice Module │────→│ Voice Bridge │────→│ OpenClaw Session │
│ CI1302       │     │ oc_voice     │     │ Xiaogua Agent    │
└─────────────┘     └──────────────┘     └────────┬─────────┘
                                                  │
┌─────────────┐                                   │
│ Feishu Msg  │───────────────────────────────────┘
│ Remote Cmd  │
└─────────────┘
                                                  │
                    ┌─────────────────────────────┼─────────────────────────────┐
                    │                             │                             │
          ┌─────────▼─────────┐       ┌───────────▼──────────┐      ┌──────────▼──────────┐
          │    Face_Emotion   │       │    person_distance   │      │     RoboOrchard     │
          │  Face Detection   │       │  Human Detection     │      │  ROS2 + CAN Bus     │
          │  Emotion Recog.   │       │  Depth Estimation    │      │  Piper Dual-arm     │
          │  Sad → Care Task  │       │  PTZ Tracking        │      │  HoloBrain VLA      │
          └─────────┬─────────┘       └───────────┬──────────┘      └──────────┬──────────┘
                    │                             │                           │
             D435 / D435i Camera              PTZ RGB-D Camera             Dual-arm Robot
```

---

## Quick Start

The following commands assume that you are in the project root directory:



```bash
pip install -r requirements.txt
```

---

### 1. Start the Emotion Recognition Module

```bash
python3 Face_Emotion/emotion.py --no-preview
```

Function description:

- Periodically detects the user's face;
- Recognizes predefined expression categories;
- When `sad` is detected, it can trigger Feishu notification, TTS broadcast, and the drink-shaking workflow.

---

### 2. Start the Voice Module Bridge

```bash
python3 Voice/oc_voice_session.py monitor --execute --session-key <your-session-key>
```

Function description:

- Listens to the CI1302 voice module serial port;
- Converts recognized voice commands into text;
- Forwards the command to an OpenClaw session;
- Executes tasks or broadcasts status according to the returned result.

Note: `session-key` is a private credential. Do not commit it to the repository.

---

### 3. Start Human Detection and Safety Alert

```bash
cd person_distance
python3 computer_security.py
```

Function description:

- Starts the PTZ camera;
- Runs YOLO human detection;
- Uses RealSense depth information to estimate human distance;
- Triggers safety alerts and stops the inference / execution chain when a person enters the risk area.

---

### 4. Start VLA Inference Services

Different tasks correspond to different VLA server scripts.

#### Object Grasping

```bash
python3 HoloBrain_ws/server_grasp_anything.py
```

#### Drink / Bottle Shaking

```bash
python3 HoloBrain_ws/server_shakebottle.py
```

#### Tableware Organization / Desktop Tidying

```bash
python3 HoloBrain_ws/server_tableware.py
```

The VLA server is responsible for:

- Receiving multi-view images, depth data, and robotic arm states;
- Calling the edge-side HBM model for action generation;
- Returning executable action segments;
- Working with the robotic arm control client to complete execution.

---

### 5. Start the Robotic Arm Control Chain

```bash
source RoboOrchard/ros2_package/install/setup.bash
```

Reset both arms:

```bash
ros2 service call /robot/left/reset_ctrl std_srvs/srv/Trigger "{}"
ros2 service call /robot/right/reset_ctrl std_srvs/srv/Trigger "{}"
```

Depending on the actual deployment, the corresponding Piper control nodes, CAN interface, ROS2 control services, and hardware interface clients also need to be started.

---

## Edge-side Models

This project contains multiple edge-side models:

| Model | Function | Deployment |
|---|---|---|
| HoloBrain VLA | Generates robotic arm action segments | ONNX → HBM, deployed on RDK S100P |
| YOLO Object Detection | Desktop object, face, and human detection | HBM / BPU inference |
| MobileFaceNet | Expression classification | ONNX / CPU-BPU mixed inference |
| Depth Estimation Logic | Human distance estimation | RealSense depth + detection boxes |

The VLA model deployment process includes:

```text
PyTorch / safetensors
→ ONNX export
→ Calibration dataset preparation
→ HBDK / OpenExplorer compilation
→ HBM model
→ RDK S100P edge-side inference
→ Real robotic arm execution verification
```

The final system uses edge-side quantized models to adapt to the computing power, memory, and runtime constraints of the RDK S100P.

---

# Safety Notes

This project focuses on low-risk home desktop manipulation tasks. The current system does not involve:

- Mobile navigation;
- Medical diagnosis;
- High-force contact tasks;
- High-risk object manipulation;
- Unsupervised high-risk robotic arm behavior.

Safety mechanisms include:

1. **Semantic Safety**
   - Rejects destructive, deceptive, unauthorized, or dangerous instructions.

2. **Target Confirmation Safety**
   - Refuses blind execution when the target is not detected.

3. **Embodiment Safety**
   - Reads the robotic arm runtime state;
   - Detects teaching mode, driver exceptions, communication exceptions, joint errors, and severe fault states;
   - Stops the task and sends alerts when an abnormal state is detected.

4. **Interaction Safety**
   - Uses the third-person RGB-D camera to detect human proximity;
   - Triggers stop with priority when risk appears;
   - Clears or pauses the pending action queue;
   - Allows recovery only after user confirmation.

**Note: The software safety chain cannot replace hardware emergency stop, low-level driver protection, or human supervision. During real tests, manual emergency stop, operation boundaries, and on-site safety monitoring are still required.**

---

## Dependencies

| Dependency | Version / Description |
|---|---|
| Ubuntu | Recommended 22.04 |
| ROS2 | Humble |
| Conda | `holo` environment |
| Python | 3.10 |
| librealsense2 | RealSense Camera SDK |
| ONNX Runtime | ONNX model inference |
| hbm_runtime / OpenExplorer | RDK S100P HBM inference |
| OpenClaw Gateway | Agent session command routing |
| CAN / Piper SDK | Piper robotic arm control |
| Feishu API | Feishu remote interaction and alerts |

---

# Performance Metrics

The following are partial test metrics from the project. Results may vary with model version, scene layout, camera frame rate, and robotic arm state.

| Category | Metric | Result |
|---|---|---|
| Edge Inference | YOLO average / P95 inference time | 2.93 / 3.69 ms |
| Edge Inference | VLA action generation time | Approx. 541.14 ms |
| Model Deployment | VLA model size | Approx. 367 MB + 36 MB |
| Safety Protection | Human proximity stop test | 10/10 |
| Remote Interaction | Feishu communication stability | 20/20 |
| Camera Stability | Four-camera continuous capture | No disconnection in 10 minutes |

---

# Development and Debugging Tips

- Before starting the system, confirm that all RealSense cameras are online.
- Before starting the robotic arms, confirm the CAN interface, power supply, and emergency stop status.
- Before running a VLA service, confirm that the HBM model path and the `text_feature` / `text_token_mask` match the current task.
- During human proximity tests, a manual emergency stop must be retained.
- Do not commit Feishu or OpenClaw session keys to GitHub.
- When multiple tasks run in parallel, prioritize the safety detection and robotic arm stop chain.

---

# FAQ

## 1. Why does the robotic arm stop when a person approaches?

This is the system's interaction safety mechanism. When a person enters the risk area, the edge-side safety process outputs a stop signal with priority and stops action dispatch, preventing the robotic arm from continuing to move.

## 2. Why does the system refuse when the user asks it to break a cup?

The OpenClaw Agent has a semantic safety barrier. Instructions involving destruction, deception, or possible harm will not be converted into robotic arm actions.

## 3. Why does the robot not execute when the requested object is not on the table?

The system first calls object detection to confirm whether the target exists. If the target is not detected, the robotic arm will not execute blindly.

## 4. Is emotion recognition a psychological diagnosis?

No. Emotion recognition is only used for lightweight companionship interaction, such as triggering voice comfort or a drink-shaking action. It is not used as a basis for medical or psychological assessment.

## 5. Is the VLA model on the RDK S100P fully equivalent to the floating-point model?

The edge-side HBM model is deployed after quantization. The focus is to verify its usability, inference speed, and real-task performance on the RDK S100P. There may be certain behavioral differences between the quantized model and the floating-point model.

---

# Competition Information

- Competition: 2026 National College Student Embedded Chip and System Design Competition
- Track: Application Track
- Project Name: Smart Home Guardian — A Dual-arm Intelligent Robot for Home Safety and Companionship
- Team: ASC-EAI
- Platform: RDK S100P
- Robot: Piper Dual-arm Robot
- Core Technologies: OpenClaw Agent, HoloBrain VLA, edge-side HBM inference, multi-view RGB-D perception, voice / Feishu interaction, safety stop

---

# License

This project is licensed under the MIT License. See the `LICENSE` file in the project root directory for details.

---

# Acknowledgements

This project uses or refers to the following open-source ecosystems and hardware platforms:

- OpenClaw Agent / Gateway
- HoloBrain VLA
- RoboOrchard
- Horizon RDK S100P / OpenExplorer
- Intel RealSense D435 / D435i
- Piper Robotic Arm SDK
- ROS2 Humble
