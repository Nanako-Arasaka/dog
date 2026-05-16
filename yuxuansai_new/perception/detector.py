from pathlib import Path

import cv2
import numpy as np
import torch

from perception.cropper import HoughCircleCropper
from perception.inference import TensorRTInference
from perception.model import Resnet18_dashboard


class DashboardCameraDetector:
    def __init__(
        self,
        model_path=None,
        device="cuda",
        use_tensorrt=True,
        engine_path=None,
        input_size=160,
        confidence_threshold=0.5,
        cropper=None,
        class_names=None,
        preprocess_mode="resize_center_crop",
        cls_confirm_window=2,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("未检测到 CUDA GPU。该程序要求使用 GPU 推理。")

        self.device = torch.device(device)
        self.use_tensorrt = bool(use_tensorrt)
        self.input_size = int(input_size)
        self.confidence_threshold = float(confidence_threshold)
        self.cropper = cropper if cropper is not None else HoughCircleCropper()
        self.preprocess_mode = str(preprocess_mode).strip().lower()

        self.cls_confirm_window = max(1, int(cls_confirm_window))
        self._cls_history = []
        self._stable_class = "unknown"

        default_class_names = ["high", "normal", "low"]
        self.class_names = list(class_names) if class_names else list(default_class_names)
        self.num_classes = len(self.class_names)

        if self.use_tensorrt:
            if engine_path is None:
                raise ValueError("use_tensorrt=True 时必须提供 TensorRT 引擎路径。")
            engine_file = Path(engine_path)
            if not engine_file.exists():
                raise FileNotFoundError(f"TensorRT 引擎不存在: {engine_file}")
            self.trt_infer = TensorRTInference(
                str(engine_file),
                device=device,
                input_size=self.input_size,
                preprocess_mode=self.preprocess_mode,
            )
            self.num_classes = self.trt_infer.output_shape[-1]
            if class_names and len(class_names) == self.num_classes:
                self.class_names = list(class_names)
            elif model_path and Path(model_path).exists():
                checkpoint = torch.load(model_path, map_location="cpu")
                if isinstance(checkpoint, dict) and "class_names" in checkpoint:
                    ckpt_class_names = list(checkpoint["class_names"])
                    if len(ckpt_class_names) == self.num_classes:
                        self.class_names = ckpt_class_names
                    elif self.num_classes == 3:
                        self.class_names = list(default_class_names)
                    else:
                        self.class_names = [f"class_{idx}" for idx in range(self.num_classes)]
                elif self.num_classes == 3:
                    self.class_names = list(default_class_names)
                else:
                    self.class_names = [f"class_{idx}" for idx in range(self.num_classes)]
            elif self.num_classes == 3:
                self.class_names = list(default_class_names)
            else:
                self.class_names = [f"class_{idx}" for idx in range(self.num_classes)]
        else:
            if model_path is None:
                raise ValueError("use_tensorrt=False 时必须提供 PyTorch 模型路径。")
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict):
                self.num_classes = int(checkpoint.get("num_classes", 3))
                ckpt_class_names = checkpoint.get("class_names")
                if class_names and len(class_names) == self.num_classes:
                    self.class_names = list(class_names)
                elif ckpt_class_names:
                    self.class_names = list(ckpt_class_names)
                else:
                    self.class_names = ["high", "normal", "low"] if self.num_classes == 3 else [
                        f"class_{idx}" for idx in range(self.num_classes)
                    ]
            else:
                self.num_classes = 3
                self.class_names = ["high", "normal", "low"]

            self.model = Resnet18_dashboard(num_classes=self.num_classes, pretrained=False, dropout=0.35).to(self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.eval()
            self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _resize_like_training(self, frame_rgb):
        if self.preprocess_mode == "resize_center_crop":
            target_short = int(round(self.input_size * 256 / 224))
            h, w = frame_rgb.shape[:2]
            scale = float(target_short) / float(min(h, w))
            resized_w = max(1, int(round(w * scale)))
            resized_h = max(1, int(round(h * scale)))
            scaled = cv2.resize(frame_rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
            x1 = max(0, (resized_w - self.input_size) // 2)
            y1 = max(0, (resized_h - self.input_size) // 2)
            cropped = scaled[y1:y1 + self.input_size, x1:x1 + self.input_size]
            if cropped.shape[0] == self.input_size and cropped.shape[1] == self.input_size:
                return cropped
            return cv2.resize(scaled, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(frame_rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)

    def preprocess_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = self._resize_like_training(frame_rgb)
        image = resized.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        image = np.transpose(image, (2, 0, 1))
        image = np.ascontiguousarray(image)
        return torch.from_numpy(image).unsqueeze(0)

    def predict(self, frame):
        crop, cx, cy = self.cropper.detect(frame)
        if crop is None:
            probabilities = np.zeros(self.num_classes, dtype=np.float32)
            return -1, "unknown", 0.0, probabilities, None, False, None, None

        if self.use_tensorrt:
            pred_class, confidence = self.trt_infer.infer(crop)
            probabilities = np.zeros(self.num_classes, dtype=np.float32)
            if 0 <= pred_class < self.num_classes:
                probabilities[pred_class] = confidence
        else:
            input_tensor = self.preprocess_frame(crop).to(self.device, non_blocking=True)
            with torch.inference_mode():
                output = self.model(input_tensor)
                probs = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probs, 1)
            pred_class = int(predicted.item())
            confidence = float(confidence.item())
            probabilities = probs.detach().cpu().numpy()[0]

        class_name = self.class_names[pred_class] if 0 <= pred_class < len(self.class_names) else "unknown"
        detected = confidence >= self.confidence_threshold
        if not detected:
            class_name = "unknown"

        self._cls_history.append(class_name)
        if len(self._cls_history) > self.cls_confirm_window:
            self._cls_history.pop(0)
        if len(self._cls_history) >= self.cls_confirm_window \
                and all(c == self._cls_history[0] for c in self._cls_history):
            self._stable_class = class_name

        return pred_class, self._stable_class, confidence, probabilities, crop, \
            detected, cx, cy


def open_camera_with_fallback(index=0):
    backends = [("CAP_V4L2", cv2.CAP_V4L2), ("CAP_ANY", cv2.CAP_ANY)]
    for name, backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                print(f"摄像头打开成功: index={index}, backend={name}")
                return cap
        cap.release()
    return None
