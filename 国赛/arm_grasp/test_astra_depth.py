#!/usr/bin/env python3
"""最小真深度验证脚本 — 直连 OpenNI2 + liborbbec, 不拉 ROS。

目的
----
astra_camera_node.py 有 5 个后端 + cv_bridge + ROS 消息往返, 排错链路太长。
这个脚本绕过 ROS, 直接调 astra_camera_node.py 真正使用的那条 OpenNI2 路径,
30 秒告诉你:
  ✓ 设备能不能枚举出来
  ✓ 深度帧能不能稳定出 (fps, 覆盖率)
  ✓ 中心点能不能采到深度 (≥30cm, <8m)
  ✓ 模拟 vision_node 的多级窗口 fallback 在你机器上够不够用

用法
----
  # 自动探测 ~/openni2/OpenNI-Linux-*/Redist:
  python3 test_astra_depth.py

  # 显式指定 (跟 astra_camera_node.py 的 OPENNI2_REDIST 参数一致):
  OPENNI2_REDIST=/home/jetson/openni2/OpenNI-Linux-Arm64-2.3.0.66/Redist \\
  LD_LIBRARY_PATH=$OPENNI2_REDIST \\
  python3 test_astra_depth.py

退出码
------
  0 = 真深度可用
  1 = 设备打不开 / 初始化失败
  2 = 设备能开但拿不到有效深度 (镜头前没东西 或 距离超范围)
  3 = 覆盖率太低 (东西太远 / 太反光 / 太透明), 抓取会反复 invalid_depth
"""

import os
import sys
import time
import glob
import numpy as np


# ─── STEP 0: 环境变量 ────────────────────────────────────────
# python openni 包用 ctypes 加载 libOpenNI2.so — 它不读 OPENNI2_REDIST,
# 只读 LD_LIBRARY_PATH 和硬编码列表 (/lib, /usr/lib, /usr/local/lib, cwd)。
# C++ 驱动的 liborbbec.so 反而是 OPENNI2_REDIST 在管。两者都得设。

REDIST = os.environ.get('OPENNI2_REDIST', '').strip()
if not REDIST:
    for base in glob.glob(os.path.expanduser('~/openni2/OpenNI-Linux-*')):
        cand = os.path.join(base, 'Redist')
        if os.path.isdir(cand):
            REDIST = cand
            break

if REDIST:
    os.environ['OPENNI2_REDIST'] = REDIST
    cur_ld = os.environ.get('LD_LIBRARY_PATH', '')
    os.environ['LD_LIBRARY_PATH'] = (
        REDIST + (':' + cur_ld if cur_ld else '')
    )

print(f'[env] OPENNI2_REDIST = {os.environ.get("OPENNI2_REDIST", "(unset)")}')
print(f'[env] LD_LIBRARY_PATH = {os.environ.get("LD_LIBRARY_PATH", "(unset)")[:200]}')


# ─── STEP 1: import openni ───────────────────────────────────
try:
    from openni import openni2
except ImportError as exc:
    print(f'\n[!!] import openni 失败: {exc}')
    print('    pip install openni')
    sys.exit(1)


# ─── STEP 2: initialize ──────────────────────────────────────
print('\n[1/4] openni2.initialize() ...')
try:
    openni2.initialize()
    print('[1/4] OK')
except Exception as exc:
    print(f'[!!] initialize 失败: {exc}')
    print('    最常见原因: OPENNI2_REDIST 没指到 Redist 目录, 或')
    print('    LD_LIBRARY_PATH 没把 Redist 加进去 (libOpenNI2.so 找不到)')
    sys.exit(1)


# ─── STEP 3: open device ─────────────────────────────────────
print('\n[2/4] Device.open_any() ...')
try:
    dev = openni2.Device.open_any()
    print('[2/4] OK')
    try:
        info = dev.get_device_info()
        print(f'      device: {info}')
    except Exception:
        pass
except Exception as exc:
    print(f'[!!] 没找到 Astra 设备: {exc}')
    print('    检查: USB 插好? 12V 通了? lsusb 看到 1a86:7523?')
    openni2.unload()
    sys.exit(1)


# ─── STEP 4: read N frames ───────────────────────────────────
print('\n[3/4] 创建并启动深度流 ...')
depth = dev.create_depth_stream()
depth.start()
print('[3/4] OK')

N = 30
print(f'\n[4/4] 读 {N} 帧 (约 1 秒) ...')

frames = []
t0 = time.time()
for _ in range(N):
    f = depth.read_frame()
    if f is None:
        continue
    arr = np.frombuffer(f.get_data(), dtype=np.uint16).reshape(
        f.get_height(), f.get_width()
    ).copy()
    frames.append(arr)
dt = time.time() - t0

depth.stop()
dev.close()
openni2.unload()

if not frames:
    print('[!!] 30 帧全部为 None — 深度流没出数据')
    sys.exit(1)

print(f'[4/4] OK, 收到 {len(frames)}/{N} 帧, {dt:.2f}s, {len(frames)/dt:.1f} fps')


# ─── STEP 5: 统计 ────────────────────────────────────────────
stack = np.stack(frames, axis=0)
h, w = stack.shape[1:]
print(f'\n每帧 shape: {h}x{w}, dtype: {stack.dtype} (16UC1, mm)')

valid_mask = (stack > 0) & (stack < 8000)
cov = valid_mask.mean(axis=(1, 2))
print(f'\n深度覆盖率 (有效像素 / 总像素):')
print(f'  min  = {cov.min()*100:7.3f}%')
print(f'  max  = {cov.max()*100:7.3f}%')
print(f'  mean = {cov.mean()*100:7.3f}%')
print(f'  ── 参考: < 0.01% = 几乎抓不到, > 0.1% = 凑合, > 1% = 健康')

last = frames[-1]
last_valid = last[valid_mask[-1]]
if len(last_valid) == 0:
    print('\n[!!] 最后一帧没有任何有效深度像素!')
    print('    镜头前 0.3~8m 内没东西? 或者镜头被遮住了?')
    sys.exit(2)

print(f'\n最后一帧有效深度 (mm):')
print(f'  min = {last_valid.min()}  median = {int(np.median(last_valid))}  '
      f'max = {last_valid.max()}')


# ─── STEP 6: 中心点 + vision_node 的多级窗口 fallback ───────
cx_pix, cy_pix = w // 2, h // 2
center_val = int(last[cy_pix, cx_pix])
print(f'\n中心像素 ({cx_pix}, {cy_pix}) = {center_val} mm '
      f'({"饱和/无回波" if center_val == 0 else "OK"})')

print(f'\n模拟 vision_node._sample_depth 多级窗口 fallback:')
hit_r = None
for r in (5, 20, 40, 60, 100):
    y1 = max(0, cy_pix - r); y2 = min(h, cy_pix + r + 1)
    x1 = max(0, cx_pix - r); x2 = min(w, cx_pix + r + 1)
    roi = last[y1:y2, x1:x2]
    v = roi[(roi > 0) & (roi < 8000)]
    if len(v) > 0:
        med = int(np.median(v))
        print(f'  r={r:3d}: {len(v):4d} 个有效像素, 中值 {med} mm')
        if hit_r is None:
            hit_r = r
    else:
        print(f'  r={r:3d}: 0 个有效像素')

if hit_r is None:
    print(f'\n[!!] r=100 都采不到深度 — 这个区域抓不到东西')
elif hit_r > 40:
    print(f'\n[!] r={hit_r} 才采到 — 物体不在中心, 或物体表面太反光')
    print('    抓取时延时会增加 (多扫几帧), 但应该能用')
else:
    print(f'\n[✓] r={hit_r} 就采到 — 抓取响应很快')


# ─── STEP 7: 简易 ASCII 可视化 ──────────────────────────────
print(f'\n深度分布 ASCII 图 ({w}x{h} 降采样到 20x15):')
print('  . = 无效, ▁▂▃▄▅▆▇ = 越深越远, 空格 = 太近 (<30cm)')
dh, dw = h // 15, w // 20
for y in range(15):
    row = ''
    for x in range(20):
        block = last[y*dh:(y+1)*dh, x*dw:(x+1)*dw]
        v = block[(block > 0) & (block < 8000)]
        if len(v) == 0:
            row += '.'
        else:
            med = np.median(v)
            if   med <  300: row += ' '
            elif med <  500: row += '▁'
            elif med <  800: row += '▂'
            elif med < 1200: row += '▃'
            elif med < 1600: row += '▄'
            elif med < 2200: row += '▅'
            elif med < 3000: row += '▆'
            elif med < 5000: row += '▇'
            else:            row += '#'
    print(f'  {row}')


# ─── STEP 8: 总结 ────────────────────────────────────────────
print(f'\n=== 验证清单 ===')
ok_coverage = cov.mean() > 0.001
ok_center = last[cy_pix, cx_pix] > 0 or hit_r is not None
ok_enough = len(last_valid) > 100

print(f'  [{"✓" if ok_coverage else "✗"}] 深度覆盖率 > 0.1% (真深度有数据)')
print(f'  [{"✓" if ok_center else "✗"}] 中心区域能采到深度 (r≤100)')
print(f'  [{"✓" if ok_enough else "✗"}] 有效像素 > 100 (有东西可抓)')

if not ok_coverage:
    print(f'\n[!] 覆盖率太低 — 物体可能太远 / 太反光 / 太透明')
    print(f'    抓取时大部分帧会 invalid_depth, 走 fixed_depth 兜底')
    sys.exit(3)
elif not ok_center:
    print(f'\n[!] 中心没深度 — 抓取目标不在视野中心')
    sys.exit(3)
else:
    print(f'\n[✓] 真深度可用, 可以关 fixed_depth 跑 vision_node')
    sys.exit(0)