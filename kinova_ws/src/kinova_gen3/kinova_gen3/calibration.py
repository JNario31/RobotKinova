import cv2
import numpy as np
import pickle

# Chessboard dimensions (inner corners)
CHESSBOARD_SIZE = (7, 7)  # Columns, Rows of INNER corners
SQUARE_SIZE = 35  # Size of each square in mm (or whatever unit you want)

# Termination criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (0,0,0), (1,0,0), (2,0,0) ... scaled by square size
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# Arrays to store object points and image points
objpoints = []  # 3D points in real world
imgpoints = []  # 2D points in image plane

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("Starting calibration...")
print(f"Looking for {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} chessboard pattern")
print("Press SPACE to capture image when pattern is detected")
print("Press 'q' to quit and compute calibration (need at least 10 images)")
print("Press 'c' to compute calibration with current images")

images_captured = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Find chessboard corners
    ret_corners, corners = cv2.findChessboardCorners(
        gray, 
        CHESSBOARD_SIZE,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    
    display_frame = frame.copy()
    
    if ret_corners:
        # Refine corner locations
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # Draw corners
        cv2.drawChessboardCorners(display_frame, CHESSBOARD_SIZE, corners_refined, ret_corners)
        
        # Show status
        cv2.putText(display_frame, "Pattern detected! Press SPACE to capture", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(display_frame, "No pattern detected", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.putText(display_frame, f"Images captured: {images_captured}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow('Calibration', display_frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord(' ') and ret_corners:
        # Capture this image
        objpoints.append(objp)
        imgpoints.append(corners_refined)
        images_captured += 1
        print(f"Captured image {images_captured}")
        
        # Flash effect
        cv2.imshow('Calibration', np.ones_like(frame) * 255)
        cv2.waitKey(100)
    
    elif key == ord('q'):
        break
    
    elif key == ord('c'):
        if images_captured >= 10:
            break
        else:
            print(f"Need at least 10 images, currently have {images_captured}")

cap.release()
cv2.destroyAllWindows()

if images_captured < 10:
    print(f"Not enough images captured ({images_captured}). Need at least 10.")
    exit()

print(f"\nComputing calibration with {images_captured} images...")

# Calibrate camera
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

if ret:
    print("\nCalibration successful!")
    print(f"\nCamera Matrix:\n{camera_matrix}")
    print(f"\nDistortion Coefficients:\n{dist_coeffs}")
    
    # Calculate reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], 
                                          camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    
    print(f"\nMean reprojection error: {mean_error / len(objpoints):.4f} pixels")
    
    # Save calibration
    calibration_data = {
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs,
        'rvecs': rvecs,
        'tvecs': tvecs,
        'reprojection_error': mean_error / len(objpoints)
    }
    
    with open('camera_calibration.pkl', 'wb') as f:
        pickle.dump(calibration_data, f)
    
    print("\nCalibration saved to 'camera_calibration.pkl'")
    
    # Test undistortion
    print("\nShowing undistortion test. Press any key to exit.")
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h)
        )
        
        # Undistort
        undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, 
                                    None, new_camera_matrix)
        
        # Show side by side
        combined = np.hstack([frame, undistorted])
        cv2.putText(combined, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, "Undistorted", (w + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Undistortion Test', combined)
        
        if cv2.waitKey(1) & 0xFF != 255:
            break
    
    cap.release()
    cv2.destroyAllWindows()
else:
    print("Calibration failed!")