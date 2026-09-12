"""
로봇 행동 판단 로직

설계 요약:
- 빨간불 감지 → 정지 (/cmd_vel 에 0 발행)
- 빨간불이 아니면(=초록불로 간주) → 진행 (Nav2가 이미 그은 웨이포인트대로 이동, 별도 명령 불필요)
- temi, 보행자는 무시 (Nav2/SLAM이 웨이포인트로 이동하며 자체 처리)
- 신호등 미감지 또는 프레임 간 판정 불안정 시 → 안전하게 정지
"""

from collections import deque, Counter
import logging

logging.basicConfig(
    filename="robot_behavior.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

IGNORED_CLASSES = ["temi", "pedestrian"]


class TrafficLightController:
    def __init__(self, history_size=5, min_confidence=0.5):
        self.state = "MOVING"
        self.history_size = history_size
        self.min_confidence = min_confidence
        self.detection_history = deque(maxlen=history_size)

    def filter_low_confidence(self, detections):
        return [d for d in detections if d["conf"] >= self.min_confidence]

    def get_stable_light_color(self, current_detections):
        filtered = self.filter_low_confidence(current_detections)
        traffic_lights = [d for d in filtered if "traffic_light" in d["class"]]

        if traffic_lights:
            best = max(traffic_lights, key=lambda x: x["conf"])
            self.detection_history.append(best["class"])
        else:
            self.detection_history.append(None)

        votes = [v for v in self.detection_history if v is not None]
        if not votes:
            return None

        most_common, count = Counter(votes).most_common(1)[0]
        if count >= (self.history_size // 2 + 1):
            return most_common
        return "UNSTABLE"

    def decide_action(self, raw_detections):
        stable_light = self.get_stable_light_color(raw_detections)

        if stable_light is None or stable_light == "UNSTABLE":
            return self._stop("신호등 미감지 또는 불안정 - 안전 정지")

        if "red" in str(stable_light):
            return self._stop("빨간불 감지")

        return self._go("초록불 감지 - 진행")

    def _stop(self, reason):
        self._log_and_print("STOP", reason)
        self.state = "STOPPED"
        return "STOP"

    def _go(self, reason):
        self._log_and_print("GO", reason)
        self.state = "MOVING"
        return "GO"

    def _log_and_print(self, action, reason):
        message = f"[{action}] {reason}"
        print(message)
        logging.info(message)


def yolo_results_to_detections(yolo_result, class_names, image, color_detector):
    detections = []
    for box in yolo_result.boxes:
        class_id = int(box.cls[0])
        class_name = class_names[class_id]

        if class_name in IGNORED_CLASSES:
            continue

        confidence = float(box.conf[0])

        if "traffic_light" in class_name:
            box_xyxy = box.xyxy[0].cpu().numpy()
            verified_color, color_conf = color_detector.detect_color_combined(image, box_xyxy)
            if verified_color == "RED":
                class_name = "traffic_light_red"
            elif verified_color == "GREEN":
                class_name = "traffic_light_green"
            else:
                continue

        detections.append({"class": class_name, "conf": confidence})
    return detections
