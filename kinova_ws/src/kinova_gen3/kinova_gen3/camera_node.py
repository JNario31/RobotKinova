import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
from kinova_gen3_interfaces.srv import GetCoords
from kinova_gen3.matrix_utils import compute_transformation, apply_transformation
import os


# --- CONSTANTS --- 
# These should be measured carefully in robot base frame (meters)
# Make sure these are sorted by x-coordinate (left to right)
BLUE_SQUARE_WORLD = [[0.075, 0.265], [0.140, 0.440], [0.205, 0.335]]

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="dOXf27URLjdeZMgyJ7en"
)

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.get_logger().info('Camera node created')

        # USE UNDISTORTED IMAGE FOR INFERENCE
        self.image_path = "/home/bruno325/RobotKinova/kinova_ws/src/kinova_gen3/kinova_gen3/undistorted_image.jpg"
        
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
        
        # Run inference on UNDISTORTED image
        result = CLIENT.infer(self.image_path, model_id="cube-color-gzmh4/14")
        preds = result['predictions']
        self.get_logger().info(f'Found {len(preds)} predictions')
        for i, pred in enumerate(preds):
            self.get_logger().info(f"Prediction {i}: {pred}")

        img = cv2.imread(self.image_path)
        
        blue_squares = []

        # Find blue squares for calibration
        for pred in preds:
            if pred['class'] == "blue":
                blue_squares.append(pred)
        
        if len(blue_squares) != 3:
            self.get_logger().error(f'Blue calibration error! Found {len(blue_squares)} blue squares, need 3')
            response.x_coords = []
            response.y_coords = []
            response.class_names = []
            return response
        
        # Extract pixel coordinates from blue square predictions
        blue_square_pixels = [[pred['x'], pred['y']] for pred in blue_squares]
        
        # CRITICAL: Sort by x-coordinate to match BLUE_SQUARE_WORLD ordering
        blue_square_pixels = sorted(blue_square_pixels, key=lambda point: point[0])
        
        self.get_logger().info(f"Blue squares (pixels, sorted): {blue_square_pixels}")
        self.get_logger().info(f"Blue squares (world): {BLUE_SQUARE_WORLD}")
        
        # Compute 2D affine transform
        matrix = self.compute_affine_from_three_points(blue_square_pixels, BLUE_SQUARE_WORLD)
        self.get_logger().info(f"Affine transformation matrix (2x3):\n{matrix}")
        
        # Transform all detected objects
        x_coords = []
        y_coords = []
        class_names = []
        
        for pred in preds:
            xp = pred['x']  # Keep as float
            yp = pred['y']
            w = int(pred['width'])
            h = int(pred['height'])
            class_name = pred['class']
            conf = pred['confidence']

            # Apply affine transform manually (2x3 matrix)
            xw = matrix[0,0]*xp + matrix[0,1]*yp + matrix[0,2]
            yw = matrix[1,0]*xp + matrix[1,1]*yp + matrix[1,2]
            
            # Skip blue calibration squares and broken blocks
            if class_name != "blue" and class_name != "broken":
                x_coords.append(float(xw))
                y_coords.append(float(yw))
                class_names.append(pred['class'])
                
                self.get_logger().info(f"{pred['class']}: Pixel({xp:.1f}, {yp:.1f}) -> World({xw:.4f}, {yw:.4f})")
                
                # Draw on image for debugging
                x1 = int(xp - w/2)
                y1 = int(yp - h/2)
                x2 = int(xp + w/2)
                y2 = int(yp + h/2)

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw center point
                cv2.circle(img, (int(xp), int(yp)), 5, (0, 0, 255), -1)

                # Draw label
                label = f"{class_name} ({xw:.3f}, {yw:.3f})"
                cv2.putText(img, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        
        # Draw blue calibration squares
        for i, blue_px in enumerate(blue_square_pixels):
            cv2.circle(img, (int(blue_px[0]), int(blue_px[1])), 8, (255, 0, 0), -1)
            cv2.putText(img, f"Blue {i}", (int(blue_px[0])+10, int(blue_px[1])),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        response.x_coords = x_coords
        response.y_coords = y_coords
        response.class_names = class_names

        cv2.imwrite("/home/bruno325/RobotKinova/kinova_ws/src/kinova_gen3/kinova_gen3/output.jpg", img)
        self.get_logger().info("Saved output.jpg with annotations")

        return response

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()