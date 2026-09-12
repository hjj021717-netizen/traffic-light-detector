# Traffic Light Detector

실제 대회 환경에서 촬영한 신호등/횡단보도 데이터로 학습한 YOLO 모델입니다.

## Classes
- traffic_light_green
- traffic_light_red
- crosswalk

## Usage
from ultralytics import YOLO
model = YOLO('best.pt')
results = model('image.jpg', conf=0.5)
results[0].show()

## Notes
클래스 순서(names)는 실제 라벨 데이터를 직접 눈으로 확인해 검증했습니다.
0: traffic_light_green
1: traffic_light_red
2: crosswalk
