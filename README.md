# Traffic Light Detector

실제 대회 환경에서 촬영한 신호등/횡단보도 데이터로 학습한 YOLO 모델과,
색상 재검증 알고리즘입니다.

## Classes
- traffic_light_green
- traffic_light_red
- crosswalk

## Files
- best.pt: 학습된 YOLO 모델
- data.yaml: 데이터셋 설정
- color_detector.py: 신호등 색상 판정 알고리즘 (RGB + HSV Hue 결합)

## Usage
from ultralytics import YOLO
from color_detector import TrafficLightColorDetector
import cv2

model = YOLO('best.pt')
detector = TrafficLightColorDetector()

image = cv2.imread('test.jpg')
results = model('test.jpg', conf=0.5)

for box in results[0].boxes:
    class_id = int(box.cls[0])
    class_name = model.names[class_id]
    if 'traffic_light' in class_name:
        box_xyxy = box.xyxy[0].cpu().numpy()
        color, confidence = detector.detect_color_combined(image, box_xyxy)
        print(color, confidence)

## Notes
클래스 순서는 실제 라벨 데이터를 직접 확인해 검증했습니다.
0: traffic_light_green
1: traffic_light_red
2: crosswalk

색상 판정은 RGB 비율 방식과 HSV Hue 방식을 함께 사용합니다.
밝은 빨간불이 흰색에 가깝게 포화되어 보이는 경우, Hue 방식이 이를 보정합니다.
