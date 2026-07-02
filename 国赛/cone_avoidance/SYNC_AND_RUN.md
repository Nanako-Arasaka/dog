# 锥桶避障同步与实机运行命令

本文档记录当前需要同步到主工程和 Jetson 上的文件，以及实机闭环运行命令。

## 1. 同步到主工程

在 Mac 上执行：

```bash
mkdir -p /Users/silencecf/code/DOG/dog/国赛/cone_avoidance/scripts
mkdir -p /Users/silencecf/code/DOG/dog/国赛/docs

cp -p /Users/silencecf/Documents/DOG/cone_avoidance/scripts/realsense_aligned_depth_web.py \
  /Users/silencecf/code/DOG/dog/国赛/cone_avoidance/scripts/realsense_aligned_depth_web.py

cp -p /Users/silencecf/Documents/DOG/cone_avoidance/scripts/cone_yolo_best.pt \
  /Users/silencecf/code/DOG/dog/国赛/cone_avoidance/scripts/cone_yolo_best.pt

cp -p /Users/silencecf/Documents/DOG/docs/README_01_rgbd_cone_perception.md \
  /Users/silencecf/code/DOG/dog/国赛/docs/README_01_rgbd_cone_perception.md

cp -p /Users/silencecf/Documents/DOG/docs/README_02_avoidance_control.md \
  /Users/silencecf/code/DOG/dog/国赛/docs/README_02_avoidance_control.md

cp -p /Users/silencecf/Documents/DOG/cone_avoidance/README.md \
  /Users/silencecf/code/DOG/dog/国赛/cone_avoidance/PERCEPTION_PLAN.md
```

## 2. GitHub 备份

```bash
cd /Users/silencecf/code/DOG/dog

git add -- 国赛/cone_avoidance/*.md \
  国赛/cone_avoidance/*.py \
  国赛/cone_avoidance/config \
  国赛/cone_avoidance/scripts \
  国赛/docs/README_01_rgbd_cone_perception.md \
  国赛/docs/README_02_avoidance_control.md

git commit -m "Add cone avoidance perception and control integration"
git push origin main
```

不要把 `.DS_Store`、`__pycache__/`、`output/`、临时部署包目录加入提交。

## 3. Jetson Dry-Run

终端 1：

```bash
cd ~/Desktop/guosai/Jetson/controller
python3 lite2_motion_receiver.py --listen-port 5005 --dry-run
```

终端 2：

```bash
cd ~/Desktop/guosai/Jetson

python3 cone_avoidance/scripts/realsense_aligned_depth_web.py \
  --host 0.0.0.0 \
  --port 8080 \
  --model cone_avoidance/scripts/cone_yolo_best.pt \
  --conf 0.45 \
  --roi 80 \
  --control-jsonl \
  --control-rate-hz 10 | \
python3 -m cone_avoidance.main_avoidance_run \
  --receiver-ip 127.0.0.1 \
  --receiver-port 5005 \
  --dry-run
```

浏览器打开：

```text
http://<jetson-ip>:8080/
```

## 4. Jetson 实机闭环

先确认有线网络：

```bash
ip addr
ip route
ping <robot_motion_host_ip>
```

终端 1，先启动 receiver：

```bash
cd ~/Desktop/guosai/Jetson/controller

python3 lite2_motion_receiver.py \
  --listen-port 5005 \
  --robot-ip <robot_motion_host_ip> \
  --robot-port <robot_motion_host_port> \
  --timeout 0.8
```

终端 2，再启动感知和避障：

```bash
cd ~/Desktop/guosai/Jetson

python3 cone_avoidance/scripts/realsense_aligned_depth_web.py \
  --host 0.0.0.0 \
  --port 8080 \
  --model cone_avoidance/scripts/cone_yolo_best.pt \
  --conf 0.45 \
  --roi 80 \
  --control-jsonl \
  --control-rate-hz 10 | \
python3 -m cone_avoidance.main_avoidance_run \
  --receiver-ip 127.0.0.1 \
  --receiver-port 5005
```

实机前必须确认：

- `vx > 0` 是否为前进。
- `wz > 0` 是否为左转。
- `--robot-ip` 是机器狗运动主机有线网 IP。
- `--receiver-ip 127.0.0.1` 只是 Jetson 本机 receiver，不是机器狗 IP。
