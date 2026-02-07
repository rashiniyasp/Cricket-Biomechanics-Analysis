import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from src.geometry import calculate_angle
from src.phase import PhaseDetector

# --- CONFIGURATION ---
VIDEO_PATH = 'inputs/SampleSideonSideBatting (1).mov'
OUTPUT_VIDEO_PATH = 'outputs/segmented_analysis.mp4'
OUTPUT_CSV_PATH = 'outputs/metrics.csv'
CONFIDENCE_THRESHOLD = 0.5

def process_video():
    print("Loading YOLOv8 Pose Model...")
    model = YOLO('yolov8n-pose.pt') 
    phase_detector = PhaseDetector()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_PATH}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    metrics_data = []
    initial_head_x = None
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_count += 1
        
        # 1. Run Inference
        results = model(frame, verbose=False)
        
        # 2. Start with the annotated frame (FULL SKELETON)
        # This gives us the full YOLO plot with bounding box and all keypoints
        annotated_frame = results[0].plot()

        # Default values
        knee_angle = 180
        head_dev = 0
        current_phase = "Stance"
        phase_color = (200, 200, 200)

        for result in results:
            keypoints = result.keypoints.data.cpu().numpy()
            if len(keypoints) > 0:
                person = keypoints[0]
                
                # Calculate Knee Angle (Left Leg: 11, 13, 15)
                if person[11][2] > CONFIDENCE_THRESHOLD and person[13][2] > CONFIDENCE_THRESHOLD and person[15][2] > CONFIDENCE_THRESHOLD:
                    knee_angle = int(calculate_angle(person[11][:2], person[13][:2], person[15][:2]))
                    
                    # Custom Draw the Knee line on the annotated_frame (OVERLAY)
                    cv2.line(annotated_frame, (int(person[11][0]), int(person[11][1])), (int(person[13][0]), int(person[13][1])), (0,255,0), 4)
                    cv2.line(annotated_frame, (int(person[13][0]), int(person[13][1])), (int(person[15][0]), int(person[15][1])), (0,255,0), 4)
                
                # Calculate Head Deviation (Nose: 0)
                if person[0][2] > CONFIDENCE_THRESHOLD:
                    head_x = person[0][0]
                    if initial_head_x is None:
                        initial_head_x = head_x
                    head_dev = abs(head_x - initial_head_x)

                # 3. DETECT PHASE
                current_phase, phase_color = phase_detector.detect_phase(knee_angle, head_dev)

        # --- DRAW UI OVERLAY ---
        # Draw the semi-transparent box at the top (on the annotated frame)
        cv2.rectangle(annotated_frame, (0, 0), (300, 110), (0, 0, 0), -1)
        
        # Text: Phase
        cv2.putText(annotated_frame, "PHASE:", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, current_phase, (120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, phase_color, 3)
        
        # Text: Metrics
        cv2.putText(annotated_frame, f"Knee Angle: {knee_angle}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(annotated_frame, f"Head Dev: {int(head_dev)}px", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Write to video
        out.write(annotated_frame)
        cv2.imshow('Segmented Analysis', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Save Logic for CSV
        metrics_data.append({
            'Frame': frame_count,
            'Phase': current_phase,
            'Knee_Angle': knee_angle,
            'Head_Dev': head_dev
        })

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    pd.DataFrame(metrics_data).to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Segmented Video saved to {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    process_video()