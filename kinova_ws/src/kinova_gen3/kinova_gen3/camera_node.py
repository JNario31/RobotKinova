import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
from kinova_gen3_interfaces.srv import GetCoords
import pickle
import threading

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="dOXf27URLjdeZMgyJ7en"
)

BLUE_SQUARE_WORLD = [[0.440, -0.140], [0.265, -0.075], [0.335, 0.205]]

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.get_logger().info('Camera node created')
        
        # Load camera calibration
        with open('/home/bruno325/RobotKinova/kinova_ws/src/kinova_gen3/kinova_gen3/camera_calibration.pkl', 'rb') as f:
            calib = pickle.load(f)
        
        self.camera_matrix = calib['camera_matrix']
        self.dist_coeffs = calib['dist_coeffs']
        
        # Create service
        self.create_service(GetCoords, "get_coords", self._handle_get_coords)
        
        # Open camera
        self.cap = cv2.VideoCapture(4)
        if not self.cap.isOpened():
            self.get_logger().error("Failed to open camera!")
            return
        
        # Current frame storage
        self.current_frame = None
        self.lock = threading.Lock()
        
        # Start camera thread
        self.camera_thread = threading.Thread(target=self.run_camera, daemon=True)
        self.camera_thread.start()
    
    def compute_affine_from_three_points(self, pixel_pts, world_pts):
        """Returns a 2x3 affine transform from 3 world→pixel correspondences."""
        pixel_pts = np.float32(pixel_pts)
        world_pts = np.float32(world_pts)
        matrix = cv2.getAffineTransform(pixel_pts, world_pts)
        return matrix
    
    def run_camera(self):
        """Continuously capture and display camera feed"""
        while rclpy.ok():
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Undistort frame
            h, w = frame.shape[:2]
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
            )
            undistorted = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, 
                                       None, new_camera_matrix)
            
            # Update current frame
            with self.lock:
                self.current_frame = undistorted.copy()
            
            # Display
            cv2.imshow('Live Camera Feed', undistorted)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
    
    def _handle_get_coords(self, request, response):
        """Process current frame and return detected coordinates"""
        self.get_logger().info('Processing frame for coordinates...')
        
        # Get current frame
        with self.lock:
            if self.current_frame is None:
                self.get_logger().error("No frame available!")
                response.x_coords = []
                response.y_coords = []
                response.class_names = []
                return response
            
            img = self.current_frame.copy()
        
        # Save for inference
        temp_path = "/tmp/inference_frame.jpg"
        cv2.imwrite(temp_path, img)
        
        # Run inference
        result = CLIENT.infer(temp_path, model_id="cube-color-gzmh4/14")
        preds = result['predictions']
        self.get_logger().info(f'Found {len(preds)} predictions')
        
        # Find blue squares
        blue_squares = [pred for pred in preds if pred['class'] == "blue"]
        
        if len(blue_squares) != 3:
            self.get_logger().error(f'Blue calibration error! Found {len(blue_squares)}/3')
            response.x_coords = []
            response.y_coords = []
            response.class_names = []
            return response
        
        # Extract and sort blue square pixels
        blue_square_pixels = [[pred['x'], pred['y']] for pred in blue_squares]
        blue_square_pixels = sorted(blue_square_pixels, key=lambda point: point[0])
        
        # Compute transformation
        matrix = self.compute_affine_from_three_points(blue_square_pixels, BLUE_SQUARE_WORLD)
        
        # Transform all detected objects
        x_coords = []
        y_coords = []
        class_names = []
        
        for pred in preds:
            xp = pred['x']
            yp = pred['y']
            class_name = pred['class']
            
            # Apply affine transform
            xw = matrix[0,0]*xp + matrix[0,1]*yp + matrix[0,2]
            yw = matrix[1,0]*xp + matrix[1,1]*yp + matrix[1,2]
            
            if class_name not in ["blue", "broken"]:
                x_coords.append(float(xw))
                y_coords.append(float(yw))
                class_names.append(class_name)
                self.get_logger().info(f"{class_name}: ({xw:.4f}, {yw:.4f})")
        
        response.x_coords = x_coords
        response.y_coords = y_coords
        response.class_names = class_names
        
        return response

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    rclpy.spin(node)
    node.cap.release()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()