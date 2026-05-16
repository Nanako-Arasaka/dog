#!/bin/bash
# 机器狗端启动脚本
cd ~/Desktop/yuxuansai
python3 start_dog.py \
  --jetson-ip 192.168.31.174 --jetson-frame-port 6006 \
  --listen-ip 0.0.0.0 --listen-port 5005 \
  --camera-index 2 --stream-width 320 --stream-height 240 --send-hz 15 \
  --font-path /usr/share/fonts/truetype/wqy/wqy-zenhei.ttc
