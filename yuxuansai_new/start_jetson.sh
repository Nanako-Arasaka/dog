#!/bin/bash
# Jetson 端启动脚本
cd ~/Desktop/yuxuansai
python3 start_jetson.py \
  --target-ip 192.168.31.XXX --target-port 5005 \
  --engine-path ./checkpoints/model_fp32_160_v3.engine \
  --input-size 160 \
  --cls-confirm-window 2
