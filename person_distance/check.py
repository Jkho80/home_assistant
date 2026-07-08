#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""读取 result.json 并输出安全检查的中文描述"""

import json
import sys

def main():
    try:
        with open("result.json", "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"无法读取 result.json: {e}")
        sys.exit(1)

    safe = data.get("safe", True)
    detections = data.get("detections", [])
    
    if safe:
        print("✅ 安全：未检测到近距离行人")
    else:
        print("⚠️ 不安全：检测到近距离行人")
        if detections:
            for i, det in enumerate(detections, 1):
                box = det.get("box", [])
                dist = det.get("distance", 0.0)
                score = det.get("score", 0.0)
                print(f"  目标{i}: 距离={dist:.2f}m, 置信度={score:.2f}, 框={box}")

if __name__ == "__main__":
    main()
