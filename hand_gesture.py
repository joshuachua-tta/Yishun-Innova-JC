import cv2
import mediapipe as mp
import time

# --- SETUP MEDIAPIPE TASKS ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Global to store AI results
latest_result = None

def result_callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

# Initialize Landmarker
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'), # Ensure file is in folder
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=result_callback,
    num_hands=1,
    min_hand_detection_confidence=0.7
)

# --- STATE TRACKING VARIABLES ---
# This prevents the "flooding" that causes lag
last_sent_command = None 

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    
    print("System Ready. Use gestures to control the robot.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # Run AI detection
        timestamp = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp)

        # Default state for this specific frame
        current_gesture = "STOP"

        # Check detection results
        if latest_result and latest_result.hand_landmarks:
            landmarks = latest_result.hand_landmarks[0]
            
            # Tip vs Knuckle Logic (Y-axis: smaller is higher)
            index_up = landmarks[8].y < landmarks[6].y
            middle_up = landmarks[12].y < landmarks[10].y
            ring_up = landmarks[16].y < landmarks[14].y
            pinky_up = landmarks[20].y < landmarks[18].y

            # 1. FORWARD: All 4 fingers up
            if index_up and middle_up and ring_up and pinky_up:
                current_gesture = "FORWARD"
            
            # 2. TURN LEFT: Only index finger up
            elif index_up and not middle_up and not ring_up:
                current_gesture = "TURN LEFT"

            #3. TURN RIGHT: Only pinky finger up
            elif not index_up and not middle_up and pinky_up:
                current_gesture = "TURN RIGHT"

            # 4. UNRECOGNIZED: (e.g., a fist) stays as "STOP"
            else:
                current_gesture = "STOP"
        else:
            # NO HAND SEEN
            current_gesture = "STOP"

        # --- EFFICIENT COMMUNICATION LOGIC ---
        # Only send the command if the gesture has CHANGED
        if current_gesture != last_sent_command:
            if current_gesture == "FORWARD":
                print(">>> ROBOT: MOVING FORWARD")
                
            elif current_gesture == "TURN LEFT":
                print(">>> ROBOT: TURNING LEFT")

            elif current_gesture == "TURN RIGHT":
                print(">>> ROBOT: TURNING RIGHT")
                
            elif current_gesture == "STOP":
                print(">>> ROBOT: STOPPING")
            
            # Update the state tracker
            last_sent_command = current_gesture

        # Display Feedback
        cv2.putText(frame, f"COMMAND: {current_gesture}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Robot Gesture Controller', frame)
        
        if cv2.waitKey(1) & 0xFF == 27: # Press ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()
