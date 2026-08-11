#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
狗端音频/语音 UDP 指令发送器原型 —— 与 lite2_motion_receiver.py 同一条 UDP 链路。

背景
----
绝影 Lite2 运动主机 <- 感知主机(Jetson) 走 UDP 192.168.1.120:43893。
运动指令已证明可用（Lite2正式运行流程.txt:351 -> struct.pack("<3i",0x21010202,0,0) 让狗起立）。
本脚本把同一框架用于"播放音频"，但 —— 狗固件是否认"播放音频"指令码未知：
  - 文档指令集里没有"播放自定义音频"的码；只有 1.3.1 语音控制指令 code=17(值1-22)，它触发的是动作。
  - UDP 简单指令格式是 3 个 int（code, value, 0），塞不下 wav 文件内容；
    除非用"带复杂数据的指令"(type=1) 传 PCM 且狗固件有解码+播放逻辑（文档未开放）。
  => 若狗只能播"预置语音"，自定义播报(A_low.wav 等)请走 Jetson 本地 aplay，
     见 scripts/check_onboard_audio.sh 确认狗自带扬声器是否连在 Jetson 音频口。

用法（在 Jetson 上跑）
----
  # 用文档里"语音控制"指令码 17(十进制)做对照试验（它触发动作,不是播文件）
  python3 lite2_audio_sender.py --code 17 --value 1

  # 盲发探测某个候选码（抓包反推到的真实码填这里）
  python3 lite2_audio_sender.py --code 0x21019001 --value 0 --repeat 3

  # 连到真实狗（默认即 192.168.1.120:43893）
  python3 lite2_audio_sender.py --code <CODE> --value <VAL>

注意
----
  抓包方法（现场确认狗是否支持音频指令）:
    sudo tcpdump -i any udp port 43893 -X
  然后用官方 app 触发"语音控制"，观察狗收到的包里的指令码，回填上面的 --code/--value。
"""
import argparse
import socket
import struct
import time

DEFAULT_IP = "192.168.1.120"
DEFAULT_PORT = 43893


def send_cmd(code: int, value: int = 0, repeat: int = 1, interval: float = 0.08,
             ip: str = DEFAULT_IP, port: int = DEFAULT_PORT) -> None:
    """复用运动指令的同一 UDP 发送框架: struct.pack("<3i", code, value, 0)。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        for i in range(repeat):
            buf = struct.pack("<3i", code, value, 0)
            sock.sendto(buf, (ip, port))
            print(f"[send] code=0x{code:08X} value={value} ({i + 1}/{repeat}) -> {ip}:{port}")
            if repeat > 1:
                time.sleep(interval)
    finally:
        sock.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Lite2 音频/语音 UDP 指令发送器原型")
    p.add_argument("--code", type=lambda x: int(x, 0), required=True,
                   help="指令码, 支持 0x 十六进制或十进制, 如 17 / 0x21019001")
    p.add_argument("--value", type=int, default=0, help="指令值 (默认 0)")
    p.add_argument("--repeat", type=int, default=1, help="重复发送次数")
    p.add_argument("--interval", type=float, default=0.08, help="重复间隔(秒)")
    p.add_argument("--ip", default=DEFAULT_IP)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = p.parse_args()
    send_cmd(args.code, args.value, args.repeat, args.interval, args.ip, args.port)


if __name__ == "__main__":
    main()
