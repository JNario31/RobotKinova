import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
from kinova_gen3_interfaces.srv import GetCoords
from kinova_gen3.matrix_utils import compute_transformation, apply_transformation
import os


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

        self.image_path = "/home/johnn31/RobotKinova/kinova_ws/src/kinova_gen3/kinova_gen3/undistorted_image.jpg"
        # Create service that provides coordinates
        self.create_service(GetCoords, "get_coords", self._handle_get_coords)
        
        # Store latest coordinates
        self.coords = []
        self.class_names = []
        
    
def _handle_get_coords(self, request, response):
    """Process image and return detected coordinates"""
    self.get_logger().info('Processing image for coordinates...')
    
    # Run inference
    result = CLIENT.infer(self.image_path, model_id="cube-color-gzmh4/14")
    preds = result['predictions']
    
    # Find blue square for calibration
    blue_square_pred = None
    for pred in preds:
        if pred['class'] == "blue":
            blue_square_pred = pred
            break
    
    if blue_square_pred is None:
        self.get_logger().error('Blue calibration square not found!')
        response.x_coords = []
        response.y_coords = []
        response.class_names = []
        return response
    
    # Get the four corners of the blue square bounding box
    x_center = blue_square_pred['x']
    y_center = blue_square_pred['y']
    width = blue_square_pred['width']
    height = blue_square_pred['height']
    
    # Calculate four corners in pixel coordinates (top-left, top-right, bottom-right, bottom-left)
    pixel_points = [
        [x_center - width/2, y_center - height/2],  # top-left
        [x_center + width/2, y_center - height/2],  # top-right
        [x_center + width/2, y_center + height/2],  # bottom-right
        [x_center - width/2, y_center + height/2]   # bottom-left
    ]
    
    # Measure your actual blue square and put the dimensions here!
    BLUE_SQUARE_SIZE = 0.05  # TODO: MEASURE THIS IN METERS!
    
    # Corresponding world coordinates (assuming square is axis-aligned)
    world_points = [
        [BLUE_SQUARE_XW - BLUE_SQUARE_SIZE/2, BLUE_SQUARE_YW - BLUE_SQUARE_SIZE/2],  # top-left
        [BLUE_SQUARE_XW + BLUE_SQUARE_SIZE/2, BLUE_SQUARE_YW - BLUE_SQUARE_SIZE/2],  # top-right
        [BLUE_SQUARE_XW + BLUE_SQUARE_SIZE/2, BLUE_SQUARE_YW + BLUE_SQUARE_SIZE/2],  # bottom-right
        [BLUE_SQUARE_XW - BLUE_SQUARE_SIZE/2, BLUE_SQUARE_YW + BLUE_SQUARE_SIZE/2]   # bottom-left
    ]
    
    # Compute transformation with 4 points
    matrix = compute_transformation(pixel_points, world_points)
    
    self.get_logger().info(f'Transformation matrix computed:\n{matrix}')
    
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

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()