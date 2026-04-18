import cv2
import numpy as np

def crop_image_with_hough_circles(image):
    global text_center_x, text_center_y, text_radius

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 灰度化
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # 高斯模糊
    # 参数：1.dp：霍夫空间分辨率，值越小分辨率越高 2.minDist：检测圆间的最小距离，如果两个圆的距离小于该值，则其中一个将被忽略
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=0.8,
                               minDist=50, minRadius=10, param2=100, maxRadius=300)  # 霍夫圆检测
    if circles is not None:
        # 如果检测到圆形，将图形坐标转换为整数
        circles = np.round(circles[0, :]).astype("int")
        # 定义一个阈值，表示两个圆心之间的最大距离，用于判断是否合并
        threshold = 10
        # 定义一个列表，用于存储合并后的图形
        merged_circles = []
        # 遍历每个圆形
        for (x1, y1, r1) in circles:
            # 定义一个标志，表示当前的圆形是否已经被合并过
            merged = False
            # 遍历已经合并过的圆形列表
            for i in range(len(merged_circles)):
                # 获取已经合并过的圆形的坐标和半径
                (x2, y2, r2) = merged_circles[i]
                # 计算两个圆心之间的距离
                distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                # 如果距离小于阈值，说明两个圆形可以合并
                if distance < threshold:
                    # 将当前的圆形和已经合并过的圆形进行平均，得到新的圆形
                    if r1 >= r2:
                        merged_circles[i] = (x1, y1, r1)
                    else:
                        merged_circles[i] = (x2, y2, r2)
                    # 设置标志为True，表示当前的圆形已经被合并过
                    merged = True
                    # 跳出循环，不再遍历其他已经合并过的圆形
                    break
            # 如果当前的圆形没有被合并过，就将它添加到已经合并过的圆形列表中
            if not merged:
                merged_circles.append((x1, y1, r1))
        # 遍历已经合并过的圆形列表，找到最大的圆，作为检测结果
        max_circle = None
        biggest_r = -1
        for (x, y, r) in merged_circles:
            if r > biggest_r:
                max_circle = (x, y, r)
                biggest_r = r

        center_x, center_y, radius = max_circle
        text_center_x = center_x
        text_center_y = center_y
        text_radius = radius

        # 画出圆形区域的轮廓
        #cv2.circle(image, (center_x, center_y), radius, (0, 255, 0), 2)
        # 图片裁剪
        cropped_img = image[
                      max(center_y - radius, 0):min(center_y + radius, image.shape[0] - 1),
                      max(center_x - radius, 0):min(center_x + radius, image.shape[1] - 1)
                      ]
        return cropped_img
    else:
        return None

def main():
    video_path = r"C:\Users\AT-austin\Desktop\The Dog\pytorch\0.9.mp4"# 视频文件路径
    output_folder= r"C:\Users\AT-austin\Desktop\The Dog\pytorch\output\high"  # 输出帧的文件夹

    cap = cv2.VideoCapture(video_path)
    frame_count = 1075

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        #圆形区域抽帧
        cropped_circle=crop_image_with_hough_circles(frame)
        if cropped_circle is not None:
          cv2.imwrite(output_folder + "/" + f'down{frame_count}.jpg', cropped_circle)
          frame_count += 1

        # cv2.imwrite(output_folder + "/" + f'{frame_count}.jpg', frame)
        # frame_count += 1

        #cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()