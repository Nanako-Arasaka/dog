import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn
from torchvision.models import resnet18, resnet34

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Resnet18_dashboard(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = resnet18(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.model(x)


class Resnet34_dashboard(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = resnet34(pretrained=True)
        self.model.fc = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        return self.model(x)


def crop_image_with_hough_circles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = None
    for param2 in (100, 85, 70):
        cand = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=0.8,
            minDist=50,
            minRadius=5,
            param2=param2,
            maxRadius=300,
        )
        if cand is not None and len(cand[0]) >= 1:
            circles = cand
            break
    if circles is None:
        return None

    circles = np.round(circles[0, :]).astype("int")
    threshold = 10
    merged_circles = []
    for (x1, y1, r1) in circles:
        merged = False
        for i in range(len(merged_circles)):
            (x2, y2, r2) = merged_circles[i]
            distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if distance < threshold:
                merged_circles[i] = (x1, y1, r1) if r1 >= r2 else (x2, y2, r2)
                merged = True
                break
        if not merged:
            merged_circles.append((x1, y1, r1))

    max_circle = max(merged_circles, key=lambda item: item[2])
    center_x, center_y, radius = max_circle
    cv2.circle(image, (center_x, center_y), radius, (0, 255, 0), 2)
    cropped_img = image[
        max(center_y - radius, 0):min(center_y + radius, image.shape[0] - 1),
        max(center_x - radius, 0):min(center_x + radius, image.shape[1] - 1),
    ]
    return cropped_img


def dial_plate_classification(cap, model, flip_flag=1):
    preprocess = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    confidence_threshold = 0.7
    sign_buffer = [None] * 50
    p_time = 0
    final_sign = None
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取视频帧。")
            break
        if flip_flag == 1:
            frame = cv2.flip(frame, 0)

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        cropped_img = crop_image_with_hough_circles(frame)
        if cropped_img is not None:
            resized_cropped_img = cv2.resize(cropped_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(resized_cropped_img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = transforms.ToTensor()(img)
            img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
            img = img.unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(img)
            _, preds = torch.max(outputs, 1)
            label = preds.item()

            sign = ""
            if label == 0:
                sign = "down"
            elif label == 1:
                sign = "normal"
            elif label == 2:
                sign = "over"

            sign_buffer.insert(0, sign)
            sign_buffer.pop()

            most_common_sign = max(set(sign_buffer), key=sign_buffer.count)
            if sign_buffer.count(most_common_sign) > len(sign_buffer) // 2:
                cv2.putText(frame, most_common_sign, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                if most_common_sign == "down":
                    final_sign = "偏低"
                    print("偏低")
                elif most_common_sign == "normal":
                    final_sign = "正常"
                    print("正常")
                elif most_common_sign == "over":
                    final_sign = "偏高"
                    print("偏高")
                if final_sign is not None:
                    return final_sign
        else:
            cv2.putText(frame, "No round found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Dashboard Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("退出系统......")
            cap.release()
            break


def run_dashboard(camera_device="/dev/video5", camera_index=5, model_path="", width=640, height=480, flip_flag=1):
    model_path = str(model_path or (Path(__file__).resolve().parent / "checkpoints" / "model_best.pth"))
    cap = cv2.VideoCapture(camera_device)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    model = Resnet18_dashboard(num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    dial_plate_classification(cap, model, flip_flag=flip_flag)


if __name__ == "__main__":
    run_dashboard()
