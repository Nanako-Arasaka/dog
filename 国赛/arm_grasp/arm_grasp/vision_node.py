#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
节点1: 视觉识别节点 (vision_node)
功能: 接收检测请求 → 识别指定颜色长条 → 计算3D抓取位姿
国赛规则: 检测红色(异常)长条, 100×50×50mm竖放

订阅: /rgbd_cam/color/image_rect_color, /rgbd_cam/depth/image_raw,
      /rgbd_cam/color/camera_info, /vision/detect_request (String),
      /inspection/all (String)
发布: /vision/grasp_pose (String), /vision/debug_image (Image)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import yaml
import os


class VisionNode(Node):

    def __init__(self):
        super().__init__('vision_node')

        # 加载配置
        config_path = self.declare_parameter('config_path', '').value
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
        else:
            cfg = {}

        obj_cfg = cfg.get('object_config', {})
        self.hsv = obj_cfg.get('hsv_ranges', {
            'red': {'lower': [0, 120, 100], 'upper': [10, 255, 255],
                    'lower2': [170, 120, 100], 'upper2': [180, 255, 255]},
            'green': {'lower': [40, 80, 80], 'upper': [85, 255, 255]}
        })
        self.obj_h = obj_cfg.get('height', 0.05)
        self.grasp_h = obj_cfg.get('grasp_height', 0.075)
        platform_cfg = cfg.get('platform_config', {})
        self.platform_h = platform_cfg.get('height', 0.5)

        # 固定台面深度反投影兜底(红条竖放在固定高台上,z 为已知常数;
        # 深度相机不可用时仍可仅靠 2D 像素 + 已知 z 反投影出 x/y)
        fd = cfg.get('fixed_depth', {})
        self.fixed_depth_enabled = bool(fd.get('enabled', False))
        self.fixed_depth_m = float(fd.get('depth_m', 0.5))

        # 相机→机械臂变换 (需实测校准)
        cam2arm = cfg.get('camera_to_arm', {})
        self.cam2arm = np.array([cam2arm.get('x', 0.1),
                                 cam2arm.get('y', 0.0),
                                 cam2arm.get('z', -0.15)])

        # 参数
        self.target_color = self.declare_parameter('target_color', 'red').value
        self.min_area = self.declare_parameter('min_area', 500).value
        self.min_conf = self.declare_parameter('min_confidence', 0.3).value
        self.color_topic = self.declare_parameter(
            'color_topic', '/rgbd_cam/color/image_rect_color').value
        self.depth_topic = self.declare_parameter(
            'depth_topic', '/rgbd_cam/depth/image_raw').value
        self.info_topic = self.declare_parameter(
            'info_topic', '/rgbd_cam/color/camera_info').value

        # ── 放置区字母识别（复用 7 类 YOLO：zone_A/B/C/D）──
        self.zone_model_path = self.declare_parameter(
            'zone_model_path', 'best_7class.pt').value
        self.zone_conf = float(self.declare_parameter('zone_confidence', 0.35).value)
        self.zone_topic = self.declare_parameter(
            'zone_topic', '/placement/recognized_zone').value

        # 状态
        self.bridge = CvBridge()
        self.cam_K = None
        self.color_img = None
        self.depth_img = None
        self.pending_request = None   # 待处理的检测请求颜色
        self.pending_zone_request = False  # 待处理的放置区字母识别请求
        self._last_z_source = 'none'  # 'depth' | 'fixed'（调试用）
        self._yolo_model = None       # 延迟加载（首次 zone 请求时）
        self._yolo_error = None

        # 订阅 — 相机
        self.create_subscription(Image, self.color_topic,
                                 self._cb_color, 10)
        self.create_subscription(Image, self.depth_topic,
                                 self._cb_depth, 10)
        self.create_subscription(CameraInfo, self.info_topic,
                                 self._cb_info, 10)

        # 订阅 — 检测请求
        self.create_subscription(String, '/vision/detect_request',
                                 self._cb_detect_req, 10)
        # 订阅 — 巡检结果(备用颜色触发)
        self.create_subscription(String, '/inspection/all',
                                 self._cb_inspection, 10)

        # 发布
        self.pub_pose = self.create_publisher(String, '/vision/grasp_pose', 10)
        self.pub_dbg = self.create_publisher(Image, '/vision/debug_image', 10)
        # 放置区字母识别结果（A/B/C/D 或 none）
        self.pub_zone = self.create_publisher(String, self.zone_topic, 10)

        # 主循环 (10Hz)
        self.create_timer(0.1, self._timer)
        self.get_logger().info('[视觉节点] 就绪，等待检测请求...')

    # ── 回调 ────────────────────────────────

    def _cb_color(self, msg):
        try:
            self.color_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except:
            pass

    def _cb_depth(self, msg):
        try:
            self.depth_img = self.bridge.imgmsg_to_cv2(msg, '16UC1')
        except:
            pass

    def _cb_info(self, msg):
        if self.cam_K is None:
            self.cam_K = np.array(msg.k).reshape(3, 3)
            self.get_logger().info(f'[视觉节点] 相机内参已加载 fx={self.cam_K[0,0]:.1f}')

    def _cb_detect_req(self, msg):
        """接收检测请求: 'red' / 'green'(长条) 或 'zone' / 'place_zone'(放置区字母)"""
        req = msg.data.strip().lower()
        if req in ('red', 'green'):
            self.pending_request = req
            self.get_logger().info(f'[视觉节点] 收到检测请求: {req}')
        elif req in ('zone', 'place_zone'):
            # 注：故意不接受裸 'place'，避免与机械臂 place 命令（'/arm/command'）
            # 字符串字面冲突；后者目前不会发到这里，但留个语义干净。
            self.pending_zone_request = True
            self.get_logger().info('[视觉节点] 收到放置区字母识别请求')
        else:
            self.get_logger().warn(f'[视觉节点] 忽略未知检测请求: {req!r}')

    def _cb_inspection(self, msg):
        """从巡检结果推断目标颜色(异常=red)"""
        # 只在没有显式请求时作为后备
        if self.pending_request is not None:
            return
        if 'abnormal' in msg.data.lower():
            self.pending_request = 'red'
            self.get_logger().info('[视觉节点] 从巡检结果触发红色检测')

    # ── 检测核心 ────────────────────────────

    def _detect(self, img, color):
        """HSV 颜色分割 + 轮廓筛选"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        r = self.hsv.get(color, {})
        m1 = cv2.inRange(hsv, np.array(r['lower']), np.array(r['upper']))
        mask = cv2.bitwise_or(mask, m1)
        if 'lower2' in r:
            m2 = cv2.inRange(hsv, np.array(r['lower2']), np.array(r['upper2']))
            mask = cv2.bitwise_or(mask, m2)

        k = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        objs = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            rect = cv2.minAreaRect(c)
            box = np.int0(cv2.boxPoints(rect))
            cx, cy = rect[0]
            w, h = rect[1]
            angle = rect[2]
            # 确保 w ≥ h (长边)
            if w < h:
                w, h = h, w
                angle += 90
            ratio = w / h if h > 0 else 0
            # 长条宽高比范围:放宽到 1.0~4.0 (2026-08-15 现场测试:实物 ratio 1.08~1.22,默认 1.3~3.5 误杀)
            if 1.0 < ratio < 4.0:
                objs.append({
                    'cx': int(cx), 'cy': int(cy),
                    'w': w, 'h': h, 'angle': angle,
                    'area': area, 'color': color, 'box': box
                })
        objs.sort(key=lambda x: x['area'], reverse=True)
        return objs

    def _sample_depth(self, cx, cy, radius=5):
        """在 (cx,cy) 周围采样深度中值，避免噪声。

        中心窗口无有效深度(0=饱和/无回波)时，逐级扩大搜索窗口重试，
        取最近的有效深度。红条中心若为饱和 0，其本体/边缘通常仍有有效值。
        OpenNI2 深度有效像素极稀疏(实测 0.08%),必须用大窗口 + 低阈值。
        """
        if self.depth_img is None:
            return 0
        h, w = self.depth_img.shape
        for r in (radius, 20, 40, 60, 100):
            x1 = max(0, cx - r)
            x2 = min(w, cx + r + 1)
            y1 = max(0, cy - r)
            y2 = min(h, cy + r + 1)
            if x2 <= x1 or y2 <= y1:
                continue
            roi = self.depth_img[y1:y2, x1:x2]
            valid = roi[(roi > 0) & (roi < 8000)]
            if len(valid) >= 1:  # OpenNI2 太稀疏，只要有 1 个有效深度就要
                self.get_logger().info(
                    f'深度采样: r={r} 有效像素={len(valid)} 中值={np.median(valid):.0f}mm '
                    f'全图有效像素={int((self.depth_img[(self.depth_img>0)&(self.depth_img<8000)]).size)}'
                )
                return np.median(valid)
        return 0

    def _to_arm_frame(self, cx, cy):
        """像素 → 相机坐标 → 机械臂基座坐标

        深度可用时用深度采样 z；深度不可用(或无效)但启用了 fixed_depth 时，
        用固定台面深度反投影(红条竖放在固定高台上,z 为已知常数)。
        """
        if self.cam_K is None:
            return None

        d = self._sample_depth(cx, cy) if self.depth_img is not None else 0
        if d == 0:
            if self.fixed_depth_enabled and self.fixed_depth_m > 0:
                z_cam = self.fixed_depth_m
                self._last_z_source = 'fixed'
            else:
                return None
        else:
            z_cam = d / 1000.0
            self._last_z_source = 'depth'

        # 像素 → 相机坐标系 (m)
        fx, fy = self.cam_K[0, 0], self.cam_K[1, 1]
        u0, v0 = self.cam_K[0, 2], self.cam_K[1, 2]
        x_cam = (cx - u0) * z_cam / fx
        y_cam = (cy - v0) * z_cam / fy

        # 相机坐标系 → 机械臂基座坐标系
        p_cam = np.array([x_cam, y_cam, z_cam])
        p_arm = p_cam + self.cam2arm
        return (p_arm[0], p_arm[1], p_arm[2])

    # ── 放置区字母识别 ────────────────────────

    def _detect_zone(self):
        """识别视野中的放置区字母(A/B/C/D)，结果发布到 /placement/recognized_zone。

        复用 7 类 YOLO(best_7class.pt: zone_A/B/C/D + gauge 三态)；
        模型延迟加载(首次 zone 请求时)，加载失败/推理异常 → 发布 'none'(FSM 会重试/兜底)。
        """
        if self.color_img is None:
            self._publish_zone('none')
            return
        if self._yolo_model is None and self._yolo_error is None:
            try:
                from ultralytics import YOLO
                self._yolo_model = YOLO(self.zone_model_path)
                self.get_logger().info(
                    f'[视觉节点] 放置区字母模型已加载: {self.zone_model_path}')
            except Exception as exc:  # noqa: BLE001
                self._yolo_error = str(exc)
                self.get_logger().error(f'[视觉节点] YOLO 加载失败: {exc}')
                self._publish_zone('none')
                return
        if self._yolo_model is None:
            self.get_logger().warn(f'[视觉节点] 字母识别不可用: {self._yolo_error}')
            self._publish_zone('none')
            return
        try:
            results = self._yolo_model.predict(
                self.color_img, imgsz=416, conf=self.zone_conf, verbose=False)
            r = results[0]
            best, best_c = None, 0.0
            if r.boxes is not None and len(r.boxes) > 0:
                names = self._yolo_model.names
                for cls_id, conf in zip(r.boxes.cls, r.boxes.conf):
                    name = names.get(int(cls_id), '').strip().upper()
                    if name.startswith('ZONE_') and len(name) > 5:
                        letter = name[-1]
                        if letter in 'ABCD' and float(conf) > best_c:
                            best, best_c = letter, float(conf)
            if best:
                self.get_logger().info(
                    f'[视觉节点] ★ 放置区字母识别: {best} conf={best_c:.2f}')
                self._publish_zone(best)
            else:
                self.get_logger().warn('[视觉节点] 未识别到放置区字母(zone_* 无命中)')
                self._publish_zone('none')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'[视觉节点] 字母识别失败: {exc}')
            self._publish_zone('none')

    def _publish_zone(self, zone: str) -> None:
        self.pub_zone.publish(String(data=zone))

    # ── 主循环 ──────────────────────────────

    def _timer(self):
        if self.pending_zone_request:
            self.pending_zone_request = False
            self._detect_zone()
        if self.color_img is None or self.pending_request is None:
            return

        color = self.pending_request
        self.pending_request = None  # 清空请求

        objs = self._detect(self.color_img, color)
        if not objs:
            self.get_logger().warn(f'[视觉节点] 未检测到{color}色长条')
            self.pub_pose.publish(String(data='none'))
            return

        # 选面积最大的
        best = objs[0]
        p_arm = self._to_arm_frame(best['cx'], best['cy'])
        if p_arm is None:
            if not self.fixed_depth_enabled:
                hint = ('深度不可用且 fixed_depth 未启用: 现场可在 grasp_config 填实测 depth_m '
                        '后置 fixed_depth.enabled: true 兜底')
            else:
                hint = '深度与 fixed_depth 均无效'
            self.get_logger().warn(f'[视觉节点] 坐标无效({hint})')
            self.pub_pose.publish(String(data='invalid_depth'))
            return

        gx, gy, gz = p_arm[0], p_arm[1], p_arm[2]
        # 抓取点在物体顶部(物体高度 + 抓取余量)
        gz_grasp = gz + self.grasp_h

        conf = min(best['area'] / 8000, 1.0)
        if conf < self.min_conf:
            self.get_logger().warn(f'[视觉节点] 置信度过低 {conf:.2f}')
            self.pub_pose.publish(String(data='low_conf'))
            return

        # 发布: arm坐标系抓取位姿 + 像素坐标 + z 来源(诊断)
        # 格式 grasp|x|y|z|angle|conf|cx|cy|z_src;task_manager 只读 0-7,追加不破坏解析
        pose = (f'grasp|{gx:.4f}|{gy:.4f}|{gz_grasp:.4f}|'
                f'{best["angle"]:.1f}|{conf:.2f}|{best["cx"]}|{best["cy"]}|{self._last_z_source}')
        self.pub_pose.publish(String(data=pose))
        self.get_logger().info(f'[视觉节点] ★ 抓取位姿(arm系): x={gx:.3f} y={gy:.3f} '
                               f'z={gz_grasp:.3f} angle={best["angle"]:.0f}° conf={conf:.2f} '
                               f'z_src={self._last_z_source} 像素({best["cx"]},{best["cy"]})')

        # 调试图像
        vis = self.color_img.copy()
        for o in objs:
            c = (0, 0, 255) if o['color'] == 'red' else (0, 255, 0)
            cv2.drawContours(vis, [o['box']], 0, c, 2)
            cv2.circle(vis, (o['cx'], o['cy']), 5, (255, 0, 0), -1)
            cv2.putText(vis, f'{o["area"]:.0f}', (o['cx'] + 10, o['cy'] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
        try:
            self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(vis, 'bgr8'))
        except:
            pass


def main():
    rclpy.init()
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
