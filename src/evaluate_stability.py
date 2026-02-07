import pandas as pd
import numpy as np

def calculate_stability_score(csv_path):
    df = pd.read_csv(csv_path)
    
    # We measure "Jitter" by calculating the difference between Frame N and Frame N+1
    # We focus on the Knee Angle because legs are often occluded by pads/nets
    
    # Calculate difference
    knee_diff = np.abs(df['Front_Knee_Angle'].diff())
    
    # Filter out massive jumps (which are detection errors, not jitter)
    # If angle jumps > 20 degrees in 1 frame, it's a "Missed Detection", not "Jitter"
    clean_diff = knee_diff[knee_diff < 20] 
    
    avg_jitter = clean_diff.mean()
    std_jitter = clean_diff.std()
    
    print("--- MODEL STABILITY EVALUATION ---")
    print(f"Metric Analyzed: Front Knee Angle")
    print(f"Average Frame-to-Frame Jitter: {avg_jitter:.2f} degrees")
    print(f"Standard Deviation: {std_jitter:.2f}")
    
    if avg_jitter < 2.0:
        print("Rating: EXCELLENT STABILITY")
    elif avg_jitter < 5.0:
        print("Rating: ACCEPTABLE (Noisy)")
    else:
        print("Rating: POOR (Needs Smoothing Filter)")

if __name__ == "__main__":
    calculate_stability_score('outputs/metrics.csv')