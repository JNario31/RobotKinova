import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
from kinova_gen3_interfaces.srv import GetCoords
from kinova_gen3.matrix_utils import apply_transformation
import os


# --- REAL-WORLD COORDINATES OF 3 BLUE SQUARES ---
BLUE_WORLD_POINTS = {
    "blue2": (-0.210, 0.350),
    "blue1": (0.075, 0.265),
    "blue3": (0.150, 0.440)
}

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="dOXf27URLjdeZMgyJ7en"
)

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.get_logger().info('Camera node created')

        self.image_path = "/home/bruno325/RobotKinova/kinova_ws/src/kinova_gen3/kinova_gen3/original_image.jpg"
        self.create_service(GetCoords, "get_coords", self._handle_get_coords)

    
    def compute_affine_from_three_points(self, pixel_pts, world_pts):
        """Returns a 2x3 affine transform from 3 world→pixel correspondences."""
        pixel_pts = np.float32(pixel_pts)
        world_pts = np.float32(world_pts)
        matrix = cv2.getAffineTransform(pixel_pts, world_pts)
        return matrix
    

    def _handle_get_coords(self, request, response):
        self.get_logger().info('Processing image for coordinates...')

        # Detect all objects
        result = CLIENT.infer(self.image_path, model_id="cube-color-gzmh4/14")
        preds = result['predictions']

        # Extract 3 blue calibration squares
        calibration_pixels = []
        calibration_world = []

        img = cv2.imread(self.image_path)

        for pred in preds:
            class_name = pred['class']
            if class_name == "blue":
                x_center = pred['x']
                y_center = pred['y']

                calibration_pixels.append([x_center, y_center])

        # Must find all three squares
        if len(calibration_pixels) != 3:
            self.get_logger().error(f"Found {len(calibration_pixels)} blue squares, expected 3!")
            response.x_coords = []
            response.y_coords = []
            response.class_names = []
            return response

        # Compute 2D affine transform
        matrix = self.compute_affine_from_three_points(calibration_pixels, calibration_world)
        self.get_logger().info(f"Affine transformation matrix (2x3):\n{matrix}")

        # Transform all detected objects
        x_coords = []
        y_coords = []
        class_names = []

        for pred in preds:
            xp = float(pred['x'])
            yp = float(pred['y'])
            
            # Apply affine transform manually (2x3 matrix)
            Xw = matrix[0,0]*xp + matrix[0,1]*yp + matrix[0,2]
            Yw = matrix[1,0]*xp + matrix[1,1]*yp + matrix[1,2]

            x_coords.append(float(Xw))
            y_coords.append(float(Yw))
            class_names.append(pred['class'])

            self.get_logger().info(f"{pred['class']}: ({Xw:.4f}, {Yw:.4f})")

        response.x_coords = x_coords
        response.y_coords = y_coords
        response.class_names = class_names

        cv2.imwrite("output.jpg", img)

        return response

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()