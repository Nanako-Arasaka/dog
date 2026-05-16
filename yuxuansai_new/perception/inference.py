import cv2
import numpy as np
import tensorrt as trt
import torch
from pathlib import Path


class TensorRTInference:
    def __init__(self, engine_path, device="cuda", input_size=160, preprocess_mode="resize"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，TensorRT 推理必须在 GPU 上运行。")

        self.device = torch.device(device)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))
        self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        self.output_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.preprocess_mode = str(preprocess_mode).strip().lower()

        model_h = self.input_shape[2] if len(self.input_shape) >= 4 else -1
        model_w = self.input_shape[3] if len(self.input_shape) >= 4 else -1
        requested_size = int(input_size)

        if model_h > 0 and model_w > 0:
            self.infer_h = model_h
            self.infer_w = model_w
        else:
            self.infer_h = requested_size
            self.infer_w = requested_size

    def preprocess(self, image_bgr):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if self.preprocess_mode == "resize_center_crop":
            target_short = int(round(max(self.infer_w, self.infer_h) * 256 / 224))
            h, w = image_rgb.shape[:2]
            scale = float(target_short) / float(min(h, w))
            resized_w = max(1, int(round(w * scale)))
            resized_h = max(1, int(round(h * scale)))
            scaled = cv2.resize(image_rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
            x1 = max(0, (resized_w - self.infer_w) // 2)
            y1 = max(0, (resized_h - self.infer_h) // 2)
            resized = scaled[y1:y1 + self.infer_h, x1:x1 + self.infer_w]
            if resized.shape[0] != self.infer_h or resized.shape[1] != self.infer_w:
                resized = cv2.resize(scaled, (self.infer_w, self.infer_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = cv2.resize(image_rgb, (self.infer_w, self.infer_h), interpolation=cv2.INTER_LINEAR)
        image = resized.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        image = np.transpose(image, (2, 0, 1))
        image = np.ascontiguousarray(image)
        if self.input_dtype == np.float16:
            image = image.astype(np.float16, copy=False)
        else:
            image = image.astype(np.float32, copy=False)
        tensor = torch.from_numpy(image).unsqueeze(0)
        return tensor.contiguous()

    def infer(self, image, return_probs=False):
        input_tensor = self.preprocess(image)
        actual_input_shape = tuple(input_tensor.shape)

        self.context.set_input_shape(self.input_name, actual_input_shape)
        resolved_input_shape = tuple(self.context.get_tensor_shape(self.input_name))
        if any(dim < 0 for dim in resolved_input_shape):
            raise RuntimeError(f"TensorRT 输入形状未解析完成: {resolved_input_shape}")
        if resolved_input_shape != actual_input_shape:
            raise RuntimeError(
                f"TensorRT 引擎输入形状不匹配: engine={resolved_input_shape}, actual={actual_input_shape}。"
            )

        output_dims = tuple(self.context.get_tensor_shape(self.output_name))
        if any(dim < 0 for dim in output_dims):
            raise RuntimeError(f"TensorRT 输出形状未解析完成: {output_dims}")

        input_tensor = input_tensor.to(self.device, non_blocking=True)
        torch_output_dtype = torch.float16 if self.output_dtype == np.float16 else torch.float32
        output_tensor = torch.empty(output_dims, dtype=torch_output_dtype, device=self.device)

        self.context.set_tensor_address(self.input_name, input_tensor.data_ptr())
        self.context.set_tensor_address(self.output_name, output_tensor.data_ptr())

        stream_handle = torch.cuda.current_stream().cuda_stream
        self.context.execute_async_v3(stream_handle)
        torch.cuda.synchronize()

        output_np = output_tensor.float().cpu().numpy()
        exp_out = np.exp(output_np - np.max(output_np, axis=-1, keepdims=True))
        probs = exp_out / np.sum(exp_out, axis=-1, keepdims=True)
        pred_class = int(np.argmax(probs, axis=-1)[0])
        confidence = float(probs[0, pred_class])
        if return_probs:
            return pred_class, confidence, probs[0]
        return pred_class, confidence
