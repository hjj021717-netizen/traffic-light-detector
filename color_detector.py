import cv2
import numpy as np


class TrafficLightColorDetector:
    """
    YOLO가 찾은 신호등 박스 안에서 빨간색 여부를 판단.
    - RGB 비율 방식과 HSV Hue 방식을 함께 사용해 안정성을 높임
      (빨간불이 밝게 포화되어 흰색에 가깝게 보이는 경우까지 대응)
    - 빨간색이 아니면 초록불로 간주
    """
    def __init__(self, red_threshold=0.15, min_bright_pixels=5, hue_ratio_threshold=0.3):
        self.red_threshold = red_threshold
        self.min_bright_pixels = min_bright_pixels
        self.hue_ratio_threshold = hue_ratio_threshold

    def detect_color(self, image, box_xyxy, top_percent=0.15):
        """RGB 비율 기반 판정 (기존 방식)"""
        x1, y1, x2, y2 = map(int, box_xyxy)
        light_region = image[y1:y2, x1:x2]

        gray = cv2.cvtColor(light_region, cv2.COLOR_BGR2GRAY)
        if gray.size == 0:
            return "UNKNOWN", 0

        threshold = np.percentile(gray, 100 - top_percent * 100)
        bright_mask = gray >= threshold
        bright_pixels = light_region[bright_mask]

        if len(bright_pixels) < self.min_bright_pixels:
            return "UNKNOWN", 0

        b = bright_pixels[:, 0].astype(float)
        g = bright_pixels[:, 1].astype(float)
        r = bright_pixels[:, 2].astype(float)
        total = r + g + b + 1e-6
        red_score = np.mean((r - np.maximum(g, b)) / total)

        if red_score >= self.red_threshold:
            return "RED", red_score
        else:
            return "GREEN", 1 - red_score

    def detect_color_by_hue(self, image, box_xyxy, top_percent=0.3):
        """HSV Hue 기반 판정 (밝기 포화에 덜 민감함)"""
        x1, y1, x2, y2 = map(int, box_xyxy)
        light_region = image[y1:y2, x1:x2]

        hsv = cv2.cvtColor(light_region, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        if v.size == 0:
            return "UNKNOWN", 0

        threshold = np.percentile(v, 100 - top_percent * 100)
        bright_mask = v >= threshold

        bright_h = h[bright_mask]

        if len(bright_h) < self.min_bright_pixels:
            return "UNKNOWN", 0

        is_red_hue = ((bright_h <= 10) | (bright_h >= 170))
        red_ratio = np.mean(is_red_hue)

        if red_ratio >= self.hue_ratio_threshold:
            return "RED", red_ratio
        else:
            return "GREEN", 1 - red_ratio

    def detect_color_combined(self, image, box_xyxy):
        """RGB 방식과 Hue 방식을 종합해 최종 판정"""
        rgb_color, rgb_conf = self.detect_color(image, box_xyxy)
        hsv_color, hsv_conf = self.detect_color_by_hue(image, box_xyxy)

        if rgb_color == hsv_color:
            return rgb_color, max(rgb_conf, hsv_conf)
        else:
            # 의견이 갈리면 Hue 방식을 우선 신뢰 (밝기 포화에 덜 민감)
            return hsv_color, hsv_conf
