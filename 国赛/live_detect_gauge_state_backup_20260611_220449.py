from ultralytics import YOLO
import cv2
import os
import time
import subprocess


# =========================
# 参数区：后续调试主要改这里
# =========================
MODEL_PATH = "/home/jetson/yolo_deploy/best.pt"
CAMERA_ID = 0
CAMERA_PATH = "/dev/video0"
CONF_THRES = 0.25
IMG_SIZE = 416
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540
PRINT_INTERVAL = 1.0

WINDOW_NAME = "Jetson YOLO Live Detect"

# 类别映射，必须与新的 best.pt 保持一致
CLASS_NAMES = {
    0: "zone_A",
    1: "zone_B",
    2: "zone_C",
    3: "zone_D",
    4: "gauge_low",
    5: "gauge_normal",
    6: "gauge_high",
}

ZONE_CLASSES = {"zone_A", "zone_B", "zone_C", "zone_D"}
GAUGE_STATUS_TEXT = {
    "gauge_low": "偏低",
    "gauge_normal": "正常",
    "gauge_high": "偏高",
}


def check_environment():
    """启动前检查模型和摄像头设备。"""
    if not os.path.exists(MODEL_PATH):
        print("[ERROR] Model file not found:")
        print(MODEL_PATH)
        return False

    if not os.path.exists(CAMERA_PATH):
        print("[ERROR] Camera device not found:")
        print(CAMERA_PATH)
        return False

    return True


def run_v4l2_command(command):
    """尝试执行 v4l2-ctl 命令；失败只警告，不中断程序。"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=2,
        )
        if result.returncode != 0:
            msg = result.stderr.strip() or result.stdout.strip()
            print(f"[WARNING] v4l2 control failed: {command}")
            if msg:
                print(f"[WARNING] {msg}")
    except Exception as exc:
        print(f"[WARNING] v4l2 control skipped: {command}")
        print(f"[WARNING] {exc}")


def configure_camera_exposure():
    """尝试关闭自动曝光并锁定曝光值。"""
    run_v4l2_command("v4l2-ctl -d /dev/video0 --set-ctrl=exposure_auto=1")
    run_v4l2_command("v4l2-ctl -d /dev/video0 --set-ctrl=exposure_absolute=120")


def open_camera():
    """使用 V4L2 打开摄像头并设置低延迟参数。"""
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    # OpenCV 层面尝试关闭自动曝光并锁定曝光；不保证每个摄像头都支持
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)

    return cap


def draw_text_with_background(frame, text, x, y, color):
    """绘制带背景文字，提升实时画面可读性。"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    padding = 4

    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size

    x = max(int(x), 0)
    y = max(int(y), text_h + padding * 2)
    top_left = (x, y - text_h - padding * 2)
    bottom_right = (x + text_w + padding * 2, y)

    cv2.rectangle(frame, top_left, bottom_right, color, -1)
    cv2.putText(
        frame,
        text,
        (x + padding, y - padding),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def draw_overlay(frame, fps, object_count, zone_name, gauge_class):
    """绘制 FPS、目标数量、推理尺寸和巡检结果。"""
    status_text = GAUGE_STATUS_TEXT.get(gauge_class or "")
    lines = [
        f"FPS: {fps:.1f}",
        f"Objects: {object_count}",
        f"imgsz: {IMG_SIZE}",
    ]
    if zone_name:
        lines.append(f"Zone: {zone_name}")
    if status_text:
        lines.append(f"Gauge: {status_text}")
    if zone_name and status_text:
        lines.append(f"Result: {zone_name} {status_text}")

    y = 32
    for line in lines:
        cv2.putText(
            frame,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 34


def predict_with_fallback(model, frame, use_half):
    """优先 half=True 推理；失败后自动回退 half=False。"""
    try:
        results = model.predict(
            frame,
            imgsz=IMG_SIZE,
            conf=CONF_THRES,
            device=0,
            half=use_half,
            verbose=False,
        )
        return results, use_half
    except Exception as exc:
        if use_half:
            print("[WARNING] YOLO half=True failed, fallback to half=False")
            print(f"[WARNING] {exc}")
            results = model.predict(
                frame,
                imgsz=IMG_SIZE,
                conf=CONF_THRES,
                device=0,
                half=False,
                verbose=False,
            )
            return results, False
        raise


def print_inspection_result(zone_name, gauge_class):
    status_text = GAUGE_STATUS_TEXT.get(gauge_class or "")
    if zone_name:
        print(f"当前区域：{zone_name}")
    if status_text:
        print(f"仪表盘状态：{status_text}")
    if zone_name and status_text:
        print(f"巡检结果：{zone_name} 区域仪表盘{status_text}")


def main():
    if not check_environment():
        return

    # 先用 v4l2-ctl 尝试锁定曝光，失败不影响后续运行
    configure_camera_exposure()

    # 加载模型，不修改模型、不重新训练
    model = YOLO(MODEL_PATH)

    cap = open_camera()
    if not cap.isOpened():
        print("[ERROR] Failed to open camera:")
        print(CAMERA_PATH)
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    prev_time = time.time()
    last_print_time = 0.0
    use_half = True

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[ERROR] Failed to read frame from camera")
                break

            # 计算整体循环 FPS，更接近实际显示刷新速度
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            # YOLO 推理，优先启用 FP16
            results, use_half = predict_with_fallback(model, frame, use_half)
            result = results[0]

            object_count = 0
            best_zone = None
            best_zone_conf = -1.0
            best_gauge = None
            best_gauge_conf = -1.0
            detected_classes = set()

            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())

                    if conf < CONF_THRES:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    class_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                    label = f"{class_name} {conf:.2f}"

                    object_count += 1
                    detected_classes.add(class_name)

                    if class_name in ZONE_CLASSES and conf > best_zone_conf:
                        best_zone = class_name
                        best_zone_conf = conf
                    elif class_name in GAUGE_STATUS_TEXT and conf > best_gauge_conf:
                        best_gauge = class_name
                        best_gauge_conf = conf

                    color = (0, 255, 0) if class_name in ZONE_CLASSES else (0, 180, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    draw_text_with_background(frame, label, x1, y1, color)

            # 限频打印，避免每帧 print 拖慢实时性
            if now - last_print_time >= PRINT_INTERVAL:
                if detected_classes:
                    names = ", ".join(sorted(detected_classes))
                    print(f"Detected: {names}")
                    print_inspection_result(best_zone, best_gauge)
                last_print_time = now

            draw_overlay(frame, fps, object_count, best_zone, best_gauge)

            # 固定显示尺寸，减少窗口自动缩放抖动
            display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            cv2.imshow(WINDOW_NAME, display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
