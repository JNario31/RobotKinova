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

BLUE_SQUARE_WORLD = [[0.075, 0.265], [0.140, 0.440], [0.205, 0.335]]

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="dOXf27URLjdeZMgyJ7en"
)

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.get_logger().info('Camera node created')

        self.image_path = "/home/bruno325/RobotKinova/kinova_ws/src/kinova_gen3/kinova_gen3/original_image.jpg"
        # Create service that provides coordinates
        self.create_service(GetCoords, "get_coords", self._handle_get_coords)
        
        # Store latest coordinates
        self.coords = []
        self.class_names = []
    
    def compute_affine_from_three_points(self, pixel_pts, world_pts):
        """Returns a 2x3 affine transform from 3 world→pixel correspondences."""
        if isinstance(pixel_pts, dict):
            pixel_pts = list(pixel_pts.values())
        if isinstance(world_pts, dict):
            world_pts = list(world_pts.values())
        pixel_pts = np.float32(pixel_pts)
        world_pts = np.float32(world_pts)
        matrix = cv2.getAffineTransform(pixel_pts, world_pts)
        return matrix
    
    def _handle_get_coords(self, request, response):
        """Process image and return detected coordinates"""
        self.get_logger().info('Processing image for coordinates...')
        
        # Run inference
        result = CLIENT.infer(self.image_path, model_id="cube-color-gzmh4/14")
        preds = result['predictions']
        self.get_logger().info(f'Found {len(preds)} predictions')
        for i, pred in enumerate(preds):
            self.get_logger().info(f"Prediction {i}: {pred}")

        img = cv2.imread(self.image_path)
        
        blue_squares = []

        # Find blue square for calibration
        blue_square_pred = None
        for pred in preds:
            if pred['class'] == "blue":
                blue_squares.append(pred)
    
        
        if len(blue_squares) != 3:
            self.get_logger().error('Blue calibration error!')
            response.x_coords = []
            response.y_coords = []
            response.class_names = []
            return response
        
        # Compute 2D affine transform
        matrix = self.compute_affine_from_three_points(blue_squares, BLUE_SQUARE_WORLD)
        self.get_logger().info(f"Affine transformation matrix (2x3):\n{matrix}")
        
        # Get the four corners of the blue square bounding box
        for blues in blue_squares:
            x_center = blue_square_pred['x']
            y_center = blue_square_pred['y']
            width = blue_square_pred['width']
            height = blue_square_pred['height']
            
        
        # Measure your actual blue square and put the dimensions here!
        BLUE_SQUARE_SIZE = 0.045  # TODO: MEASURE THIS IN METERS!
        
        
        # Transform all detected objects
        x_coords = []
        y_coords = []
        class_names = []
        
        for pred in preds:
            xp = int(pred['x'])
            yp = int(pred['y'])
            w = int(pred['width'])
            h = int(pred['height'])
            class_name = pred['class']
            conf = pred['confidence']

            # Apply affine transform manually (2x3 matrix)
            xw = matrix[0,0]*xp + matrix[0,1]*yp + matrix[0,2]
            yw = matrix[1,0]*xp + matrix[1,1]*yp + matrix[1,2]
            
            if(class_name != "blue" or class_name != "broken"):
                x_coords.append(float(xw))
                y_coords.append(float(yw))
                class_names.append(pred['class'])
                
                self.get_logger().info(f"{pred['class']}: ({xw:.4f}, {yw:.4f})")
                
                x1 = int(xp - w/2)
                y1 = int(yp - h/2)
                x2 = int(xp + w/2)
                y2 = int(yp + h/2)

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label
                label = f"{class_name} ({conf:.2f})"
                cv2.putText(img, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        
        response.x_coords = x_coords
        response.y_coords = y_coords
        response.class_names = class_names

        cv2.imwrite("output.jpg",img)


        return response

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()