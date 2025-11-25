import cv2
import pickle
import numpy as np

with open('camera_calibration.pkl', 'rb') as f:
    calib = pickle.load(f)

camera_matrix = calib['camera_matrix']
dist_coeffs = calib['dist_coeffs']

# Test undistortion
print("\nShowing undistortion test. Press SPACE to save image, 'q' to exit.")
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
    
    hsv = cv2.cvtColor(undistorted, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv)

    # Boost saturation (colors)
    s_channel = cv2.add(s_channel, 50)   # +50 saturation
    s_channel = cv2.min(s_channel, 255)

    # Boost brightness
    v_channel = cv2.add(v_channel, 20)   # +20 brightness
    v_channel = cv2.min(v_channel, 255)

    # Merge and convert back to BGR
    hsv_boosted = cv2.merge((h_channel, s_channel, v_channel))
    enhanced = cv2.cvtColor(hsv_boosted, cv2.COLOR_HSV2BGR)

    # Optional contrast boost (linear)
    alpha = 1.2   # Contrast factor
    beta = 0      # Additional brightness (0 = unchanged)
    enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)

    # Show side by side
    combined = np.hstack([frame, enhanced])
    cv2.putText(combined, "Original", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined, "Undistorted", (w + 10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined, "Press SPACE to save, 'q' to quit", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow('Undistortion Test', combined)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # SPACE key
        cv2.imwrite('undistorted_image.jpg', enhanced)
        cv2.imwrite('original_image.jpg', frame)
        print("Saved undistorted_image.jpg and original_image.jpg")
        
        # Flash effect
        cv2.imshow('Undistortion Test', np.ones_like(combined) * 255)
        cv2.waitKey(100)
        
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()