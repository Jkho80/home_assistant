import json

with open("command.json", "w") as f:
    json.dump({"return": True}, f)


print("已发送结束检查命令。")
