# -*- coding: utf-8 -*-
from pathlib import Path
import time

import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt


TRT_ENGINE_PATH = str(Path(__file__).resolve().parent / "checkpoints" / "resnet18_dashboard.trt")


def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0)
    return image.astype(np.float32)


class TRTModel:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.input_shape = (1, 3, 224, 224)
        self.output_shape = (1, 3)
        self.d_input = cuda.mem_alloc(int(np.prod(self.input_shape) * 4))
        self.d_output = cuda.mem_alloc(int(3 * 4))
        self.stream = cuda.Stream()

    def infer(self, input_np):
        cuda.memcpy_htod_async(self.d_input, input_np, self.stream)
        self.context.execute_async_v2(
            bindings=[int(self.d_input), int(self.d_output)],
            stream_handle=self.stream.handle,
        )
        output = np.empty(self.output_shape, dtype=np.float32)
        cuda.memcpy_dtoh_async(output, self.d_output, self.stream)
        self.stream.synchronize()
        return output


def crop_image_with_hough_circles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=0.8, minDist=50, minRadius=10, param2=100, maxRadius=300)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        threshold = 10
        merged_circles = []
        for (x1, y1, r1) in circles:
            merged = False
            for i in range(len(merged_circles)):
                (x2, y2, r2) = merged_circles[i]
                distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                if distance < threshold:
                    if r1 >= r2:
                        merged_circles[i] = (x1, y1, r1)
                    else:
                        merged_circles[i] = (x2, y2, r2)
                    merged = True
                    break
            if not merged:
                merged_circles.append((x1, y1, r1))

        max_circle = None
        biggest_r = -1
        for (x, y, r) in merged_circles:
            if r > biggest_r:
                max_circle = (x, y, r)
                biggest_r = r

        center_x, center_y, radius = max_circle
        cv2.circle(image, (center_x, center_y), radius, (0, 255, 0), 2)
        cropped_img = image[
            max(center_y - radius, 0):min(center_y + radius, image.shape[0] - 1),
            max(center_x - radius, 0):min(center_x + radius, image.shape[1] - 1),
        ]
        return cropped_img
    else:
        return None


def dial_plate_classification(cap, trt_model, flag=1):
    sign_buffer = [None] * 50
    p_time = 0
    labels = ["down", "normal", "over"]

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取视频帧。")
            break

        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time

        frame_for_circle = frame.copy()
        cropped_img = crop_image_with_hough_circles(frame_for_circle)
        most_common_sign = None
        if cropped_img is not None:
            resized_cropped_img = cv2.resize(
                cropped_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR
            )
            input_tensor = preprocess_image(resized_cropped_img)
            output = trt_model.infer(input_tensor.copy())
            label = int(np.argmax(output))
            sign = labels[label]
            sign_buffer.insert(0, sign)
            sign_buffer.pop()
            most_common_sign = max(set(sign_buffer), key=sign_buffer.count)
            if sign_buffer.count(most_common_sign) > len(sign_buffer) // 2:
                if most_common_sign == "down":
                    print("偏低")
                elif most_common_sign == "normal":
                    print("正常")
                elif most_common_sign == "over":
                    print("偏高")

        flipped_frame = cv2.flip(frame_for_circle, 1)
        cv2.putText(
            flipped_frame,
            f"FPS: {int(fps)}",
            (10, 30),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (255, 0, 255),
            2,
        )
        if most_common_sign is not None:
            cv2.putText(
                flipped_frame,
                most_common_sign,
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                flipped_frame,
                "No round found",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        cv2.imshow("frame", flipped_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("退出系统......")
            cap.release()
            break


def run_dashboard(camera_device="/dev/video2", camera_index=2, engine_path="", width=640, height=480):
    engine_path = str(engine_path or TRT_ENGINE_PATH)
    cap = cv2.VideoCapture(camera_device) if camera_device else cv2.VideoCapture(camera_index)
    if camera_device and not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    trt_model = TRTModel(engine_path)
    dial_plate_classification(cap, trt_model, 1)
    cap.release()
    cv2.destroyAllWindows()
    return "Python 任务执行完毕"


if __name__ == "__main__":
    run_dashboard(camera_device="/dev/video5", camera_index=5)
