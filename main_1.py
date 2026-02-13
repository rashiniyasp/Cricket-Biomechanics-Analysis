import cv2
import numpy as np
from ultralytics import YOLO
from src.geometry import calculate_angle

# --- CONFIGURATION ---
VIDEO_PATH = 'inputs/SampleSideonSideBatting (1).mov' 
OUTPUT_PATH = 'outputs/output_analysis.mp4'
CONFIDENCE_THRESHOLD = 0.5  # Critical for the Net/Mesh issue

def process_video():
    # 1. Load the Model
    # 'yolov8n-pose.pt' is nano (fast). Use 'yolov8m-pose.pt' (medium) for better accuracy if you have a GPU.
    print("Loading YOLOv8 Pose Model...")
    model = YOLO('yolov8n-pose.pt') 

    # 2. Setup Video Capture
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_PATH}")
        return

    # 3. Setup Video Writer
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        
        # 4. Run Inference
        # We assume one person (the batsman) is the main focus
        results = model(frame, verbose=False)
        
        # 5. Extract Keypoints
        for result in results:
            # result.keypoints.data is a tensor of shape (1, 17, 3) -> (Batch, Joints, [x,y,conf])
            keypoints = result.keypoints.data.cpu().numpy()
            
            if len(keypoints) > 0:
                person_kps = keypoints[0] # Take the first detected person
                
                # --- VISUALIZATION LOGIC ---
                # We interpret keypoints based on COCO format:
                # 5: L-Shoulder, 7: L-Elbow, 9: L-Wrist
                # 11: L-Hip, 13: L-Knee, 15: L-Ankle
                # (Right side is 6, 8, 10, 12, 14, 16)
                
                # We'll calculate Left Knee for now (Hip-Knee-Ankle) indices: 11, 13, 15
                
                # Get coordinates and confidence
                hip = person_kps[11][:2]
                knee = person_kps[13][:2]
                ankle = person_kps[15][:2]
                
                hip_conf = person_kps[11][2]
                knee_conf = person_kps[13][2]
                ankle_conf = person_kps[15][2]

                # --- THE "ADVANCED" CHECK ---
                # This checks if the Net mesh blocked the view. 
                # If confidence is low, we SKIP drawing to avoid "jittery" garbage data.
                if hip_conf > CONFIDENCE_THRESHOLD and knee_conf > CONFIDENCE_THRESHOLD and ankle_conf > CONFIDENCE_THRESHOLD:
                    
                    # Calculate Metric
                    angle = calculate_angle(hip, knee, ankle)
                    
                    # Draw Overlay
                    cv2.line(frame, (int(hip[0]), int(hip[1])), (int(knee[0]), int(knee[1])), (0, 255, 0), 3)
                    cv2.line(frame, (int(knee[0]), int(knee[1])), (int(ankle[0]), int(ankle[1])), (0, 255, 0), 3)
                    cv2.circle(frame, (int(knee[0]), int(knee[1])), 10, (0, 0, 255), -1)
                    
                    # Put Text
                    cv2.putText(frame, f'Knee Angle: {int(angle)}', 
                                (int(knee[0]) + 20, int(knee[1])), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw the standard YOLO skeleton on top for reference
        annotated_frame = results[0].plot()
        
        # Write to file
        out.write(annotated_frame)
        
        # Optional: Show on screen (Press Q to quit)
        cv2.imshow('Analysis', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processing complete. Video saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    process_video()
