"""
실제 로봇(Jetson)에서 실행되는 ROS2 노드.
카메라 프레임을 받아 YOLO + 색깔 판정 알고리즘으로 신호등을 인식하고,
TrafficLightController의 판단에 따라 /cmd_vel로 정지/진행을 제어한다.

정지: 모든 축이 0인 Twist 발행
진행: 별도 명령 없음 (Nav2가 이미 웨이포인트로 이동 중이므로 그대로 둠)
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ultralytics import YOLO
from color_detector import TrafficLightColorDetector
from behavior import TrafficLightController, yolo_results_to_detections


class TrafficLightNode(Node):
    def __init__(self):
        super().__init__("traffic_light_controller")

        self.model = YOLO("best.pt")
        self.class_names = self.model.names
        self.color_detector = TrafficLightColorDetector()
        self.controller = TrafficLightController()
        self.bridge = CvBridge()

        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # 카메라 영상 토픽 구독
        self.image_sub = self.create_subscription(
            Image, "/frontvideostream", self.on_image, 10
        )

        self.get_logger().info("TrafficLightNode 시작. 클래스 순서: %s" % str(self.class_names))

    def on_image(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        results = self.model(image, conf=0.5, verbose=False)
        detections = yolo_results_to_detections(
            results[0], self.class_names, image, self.color_detector
        )
        action = self.controller.decide_action(detections)

        if action == "STOP":
            self.publish_stop()
        # action == "GO" 인 경우: Nav2가 이미 이동 중이므로 별도 발행 없음

    def publish_stop(self):
        stop_msg = Twist()  # 모든 값 기본 0
        self.cmd_vel_pub.publish(stop_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
