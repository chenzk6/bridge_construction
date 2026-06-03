import cv2
import numpy as np


def detect_object_pixel(frame):
    """
    返回: pixel, debug, mask
    pixel: (u, v) 或 None
    """
    debug = frame.copy()

    # 1) 预处理
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # 2) 黄色阈值（先用宽范围，后续再收紧）
    lower_yellow = np.array([15, 50, 50], dtype=np.uint8)
    upper_yellow = np.array([45, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 3) 形态学去噪
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    # 4) 轮廓筛选
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    target = None
    max_area = 0.0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 1500:  # 先放宽，避免漏检
            continue

        rect = cv2.minAreaRect(c)
        (_, _), (w, h), _ = rect
        if w < 1 or h < 1:
            continue

        ratio = max(w, h) / min(w, h)
        # 10x5cm 木块，长宽比约2，先给宽一点
        if not (1.3 <= ratio <= 3.5):
            continue

        if area > max_area:
            max_area = area
            target = c

    pixel = None
    if target is not None:
        M = cv2.moments(target)
        if M["m00"] > 1e-6:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            pixel = (cx, cy)

        box = cv2.boxPoints(cv2.minAreaRect(target)).astype(np.int32)
        cv2.polylines(debug, [box], True, (0, 255, 0), 2)
        if pixel is not None:
            cv2.circle(debug, pixel, 5, (0, 0, 255), -1)

    return pixel, debug, mask