#!/bin/bash
# 摇瓶子工作流 - 由 rdk_emotion_3s_trigger.py 通过 Popen 后台调用
# 检测到 sad 后异步执行，不随主进程退出而终止

export PATH="$HOME/bin:$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export HOME="$HOME"

LOG=/tmp/emotion_workflow.log
echo "[$(date '+%H:%M:%S')] === 摇瓶子工作流启动 ===" >> "$LOG"



# ① 停其他 VLA 服务
for svc in graspanything organizetableware; do
    oc_armctl "$svc" stop >> "$LOG" 2>&1
    echo "[$(date '+%H:%M:%S')] stop $svc done" >> "$LOG"
done

# ②启动安全检测（先清理残留的返回指令，避免新进程立即退出）
echo "[$(date '+%H:%M:%S')] reset command.json & security start..." >> "$LOG"
echo '{"return": false}' > ~/person_distance/command.json
TMUX="" oc_armctl security computer start >> "$LOG" 2>&1
if [ $? -ne 0 ]; then
    echo "[$(date '+%H:%M:%S')] ❌ 安全检测启动失败" >> "$LOG"
    exit 1
fi

# ③ 启动 shakebottle 服务端
echo "[$(date '+%H:%M:%S')] shakebottle start..." >> "$LOG"
oc_armctl shakebottle start >> "$LOG" 2>&1
if [ $? -ne 0 ]; then
    echo "[$(date '+%H:%M:%S')] ❌ shakebottle 启动失败" >> "$LOG"
    exit 1
fi

# ④ 启动客户端（异步）
# 注意：start_async.sh 内部的 tmuxp load 已改为 load -y 避免交互提示
echo "[$(date '+%H:%M:%S')] client async..." >> "$LOG"
oc_armctl client async >> "$LOG" 2>&1
if [ $? -ne 0 ]; then
    echo "[$(date '+%H:%M:%S')] ❌ client 启动失败" >> "$LOG"
    exit 1
fi


# 5.启用推理
echo "[$(date '+%H:%M:%S')] inference enable..." >> "$LOG"
TMUX="" oc_armctl inference enable >> "$LOG" 2>&1
if [ $? -ne 0 ]; then
    echo "[$(date '+%H:%M:%S')] ❌ 推理启动失败" >> "$LOG"
    exit 1
fi

echo "[$(date '+%H:%M:%S')] ✅ 摇瓶子工作流已完成" >> "$LOG"
