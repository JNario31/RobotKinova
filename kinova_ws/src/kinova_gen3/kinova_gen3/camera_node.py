import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
from kinova_gen3_interfaces.srv import GetCoords
from kinova_gen3.matrix_utils import compute_transformation, apply_transformation
import pickle



# --- CONSTANTS --- 
BLUE_SQUARE_XW = 0.075
BLUE_SQUARE_YW = 0.265

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="dOXf27URLjdeZMgyJ7en"
)

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.get_logger().info('Camera node created')
        
        # Create service that provides coordinates
        self.create_service(GetCoords, "get_coords", self._handle_get_coords)
        
        # Store latest coordinates
        self.coords = []
        self.class_names = []
        
    
    def _handle_get_coords(self, request, response):
        """Process image and return detected coordinates"""
        self.get_logger().info('Processing image for coordinates...')
        
        # Run inference
        result = CLIENT.infer("undistorted_image.jpg", model_id="cube-color-gzmh4/14")
        preds = result['predictions']
        
        # Find blue square for calibration
        blue_square_xp, blue_square_yp = None, None
        for pred in preds:
            if pred['class'] == "blue":
                blue_square_xp = int(pred['x'])
                blue_square_yp = int(pred['y'])
                break
        
        if blue_square_xp is None:
            self.get_logger().error('Blue calibration square not found!')
            response.x_coords = []
            response.y_coords = []
            response.class_names = []
            return response
        
        # Compute transformation (you'll need your matrix_utils here)
        pixel_points = [[blue_square_xp, blue_square_yp]]
        world_points = [[BLUE_SQUARE_XW, BLUE_SQUARE_YW]]
        matrix = compute_transformation(pixel_points, world_points)
        
        # Transform all detected objects
        x_coords = []
        y_coords = []
        class_names = []
        
        for pred in preds:
            xp = int(pred['x'])
            yp = int(pred['y'])
            xw, yw = apply_transformation(matrix, (xp, yp))
            
            x_coords.append(float(xw))
            y_coords.append(float(yw))
            class_names.append(pred['class'])
            
            self.get_logger().info(f"{pred['class']}: ({xw:.4f}, {yw:.4f})")
        
        response.x_coords = x_coords
        response.y_coords = y_coords
        response.class_names = class_names
        return response

        # response.x_coords = [0.1, 0.2, 0.3]
        # response.y_coords = [0.15, 0.25, 0.35]
        # response.class_names = ['red', 'blue', 'green']
        
        return response

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()