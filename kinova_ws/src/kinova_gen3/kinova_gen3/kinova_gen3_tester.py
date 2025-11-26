from kinova_gen3_interfaces.srv import Status, SetGripper, GetGripper, SetJoints, GetJoints, GetTool, SetTool, GetCoords
import rclpy
from rclpy.node import Node
import time
import pickle
import cv2


def do_home(node, home):
    z = Status.Request()
    future = home.call_async(z)
    rclpy.spin_until_future_complete(node, future)
    print(f"Home returns {future.result()}")
    return future.result().status

def do_set_gripper(node, set_gripper, v):
    """Set the gripper"""
    z = SetGripper.Request()
    z.value = v
    future = set_gripper.call_async(z)
    rclpy.spin_until_future_complete(node, future)
    print(f"SetGripper returns {future.result()}")
    return future.result().status

def do_get_gripper(node, get_gripper):
    """Get the current gripper setting"""
    z = GetGripper.Request()
    future = get_gripper.call_async(z)
    rclpy.spin_until_future_complete(node, future)
    print(f"GetGripper returns {future.result()}")
    return future.result().value

def do_get_tool(node, get_tool):
    z = GetTool.Request()
    future = get_tool.call_async(z)
    rclpy.spin_until_future_complete(node, future)
    print(f"GetTool returns {future.result()}")
    q = future.result()
    return q.x, q.y, q.z, q.theta_x, q.theta_y, q.theta_z

def do_set_tool(node, set_tool, x, y, z, theta_x, theta_y, theta_z):
    t = SetTool.Request()
    t.x = float(x)
    t.y = float(y)
    t.z = float(z)
    t.theta_x = float(theta_x)
    t.theta_y = float(theta_y)
    t.theta_z = float(theta_z)
    print(f"Request built {t}")
    future = set_tool.call_async(t)
    rclpy.spin_until_future_complete(node, future)
    print(f"SetTool returns {future.result()}")
    return future.result().status

def do_get_joints(node, get_joints):
    z = GetJoints.Request()
    future = get_joints.call_async(z)
    rclpy.spin_until_future_complete(node, future)
    print(f"GetJoints returns {future.result()}")
    return future.result().joints

def do_set_joints(node, set_joints, v):
    z = SetJoints.Request(joints=v)
    print(f"Request built {z}")
    future = set_joints.call_async(z)
    rclpy.spin_until_future_complete(node, future)
    print(f"SetJoints returns {future.result()}")
    return future.result().status

def do_get_coords(node, get_coords):
    z = GetCoords.Request()
    future = get_coords.call_async(z)
    rclpy.spin_until_future_complete(node, future)
    result = future.result()
    coords = list(zip(result.x_coords, result.y_coords, result.class_names))
    print(f"GetCoords returns {coords}")
    return coords

# For block stacking
def pick_block(node, set_tool, set_gripper, x, y, z, approach_height):

    # Convert from cm to m for API
    x = x 
    y = y 
    z = z 
    approach_height = approach_height

    # Open gripper
    do_set_gripper(node, set_gripper, 0.0)  # Open gripper
    time.sleep(1.5)

    # Move above block
    do_set_tool(node, set_tool, x, y, z + approach_height, 180.0, 0.0, 180.0)  # Move above block
    time.sleep(1.5)

    # Move down to block
    do_set_tool(node, set_tool, x, y, z, 180.0, 0.0, 180.0)  # Move down to block
    time.sleep(1.5)

    # Close gripper (adjust 1.0 based on block size)
    do_set_gripper(node, set_gripper, 1.0)
    time.sleep(1.5)

    # Lift Block
    do_set_tool(node, set_tool, x, y, z + approach_height, 180.0, 0.0, 180.0) 
    time.sleep(1.5)

    return True

def picture_postion(node, set_tool):
    do_set_tool(node, set_tool,0.1, 0.3, 0.4, 180.0,0.0,180.0)
    take_picture()
    time.sleep(1.5)

def take_picture():
    with open('camera_calibration.pkl', 'rb') as f:
        calib = pickle.load(f)

    camera_matrix = calib['camera_matrix']
    dist_coeffs = calib['dist_coeffs']

    cap = cv2.videoCapture(0)

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Failed to capture image from Camera.")
    
    h, w = frame.shape[:2]

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
        
    # Undistort
    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, 
                                None, new_camera_matrix)    
    
    cv2.imwrite('undistorted_image.jpg', undistorted)
    cv2.imwrite('original_image.jpg', frame)

    cap.release()
    print("Saved undistorted_image.jpg and original_image.jpg")
    

    

def place_block(node, set_tool, set_gripper, x, y, z, approach_height=15.0):

    # Convert from cm to m for API
    x = x 
    y = y
    z = z
    approach_height = approach_height

    # Move to position above target
    do_set_tool(node, set_tool, x, y, z + approach_height, 180.0, 0.0, 180.0)  # Move above block
    time.sleep(1.5)

    # Move down to target
    do_set_tool(node, set_tool, x, y, z, 180.0, 0.0, 180.0)  # Move down to block
    time.sleep(1.5)

    # Open gripper 
    do_set_gripper(node, set_gripper, 0.0)
    time.sleep(1.5)

    # move up
    do_set_tool(node, set_tool, x, y, z + approach_height, 180.0, 0.0, 180.0) 
    time.sleep(1.5)

    return True

def stack_blocks(node, set_tool, home, set_gripper, coords, n_blocks, x, y, z):

        # Pickup location configuration
        pickup_z = 0.0
        n = n_blocks

        # Block height
        block_height = 0.05

        # Place configuration
        place_x = x
        place_y = y
        place_z = z

        #Approach height
        approach_height = 0.15

        for i in range(n):
            x = coords[i][0]
            y = coords[i][1]

            # if y > 0.2:
            #     y = y - 0.2
            # elif y > 0 and y <= 0.2:
            #     y = y - 0.1
            # elif y < 0 and y >= -0.2:
            #     y = y + 0.1
            # elif y < -0.2:
            #     y = y + 0.2
                
            pick_block(node, set_tool, set_gripper, 
                    y, x - 0.15, pickup_z, 
                    approach_height=approach_height)
            place_block(node, set_tool, set_gripper, 
                    place_x, place_y, place_z + i * block_height, 
                    approach_height=approach_height)
            time.sleep(1.5)
        


def main():
    rclpy.init(args=None)
    node = Node('dummy')

    get_coords = node.create_client(GetCoords, "/get_coords")
    while not get_coords.wait_for_service(timeout_sec=1.0):
        node.get_logger().info('Waiting for get_coords')

    get_tool = node.create_client(GetTool, "/get_tool")
    while not get_tool.wait_for_service(timeout_sec=1.0):
        node.get_logger().info('Waiting for get_tool')

    set_tool = node.create_client(SetTool, "/set_tool")
    while not set_tool.wait_for_service(timeout_sec=1.0):
        node.get_logger().info('Waiting for set_tool')

    get_joints = node.create_client(GetJoints, "/get_joints")
    while not get_joints.wait_for_service(timeout_sec=1.0):
        node.get_logger().info('Waiting for get_joints')

    set_joints = node.create_client(SetJoints, "/set_joints")
    while not set_joints.wait_for_service(timeout_sec=1.0):
        node.get_logger().info('Waiting for set_joints')

    set_gripper = node.create_client(SetGripper, "/set_gripper")
    while not set_gripper.wait_for_service(timeout_sec=1.0):
        node.get_logger().info('Waiting for set_gripper')

    get_gripper = node.create_client(GetGripper, "/get_gripper")
    while not get_gripper.wait_for_service(timeout_sec=1.0):
        node.get_logger().info('Waiting for get_gripper')

    home = node.create_client(Status, "/home")
    while not home.wait_for_service(timeout_sec=1.0):
        node.get_logger().info('Waiting for home')

    coords = do_get_coords(node, get_coords)

    class_names = [coord[2] for coord in coords]
    unique_classes = set(class_names)
    picture_postion(node, set_tool)

   # Define base end coordinates
    base_x = 0.5  # Starting x position
    base_y = 0.5  # Fixed y position
    base_z = 0.1  # Fixed z position
    x_increment = -0.10  # 5cm increment for each color (0.05m = 5cm)
   
    for i, color in enumerate(unique_classes):
         picture_postion(node, set_tool)
         # Filter coordinates for this specific color
         color_coords = [coord for coord in coords if coord[2] == color]
         n_blocks = len(color_coords)
    
         # Calculate end position for this color
         end_x = base_x + (i * x_increment)
         end_y = base_y
         end_z = base_z

         print(f"Stacking color {len(color_coords)} {color} block(s) at ({end_x:.3f}, {end_y:.3f}, {end_z:.3f})")
    
         stack_blocks(node, set_tool, home, set_gripper, color_coords, n_blocks, end_x, end_y, end_z)
        



if __name__ == '__main__':
    main()
 