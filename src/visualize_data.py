import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load Data
csv_path = 'outputs/metrics.csv'
df = pd.read_csv(csv_path)

# Interpolate missing data (NaN) to make lines smooth
# This handles the frames where the net blocked the view
df = df.interpolate()

# Set up the dashboard
plt.figure(figsize=(12, 10))
plt.suptitle('Cricket Biomechanics Analysis (Batsman)', fontsize=16)

# 1. Front Knee Angle Analysis
plt.subplot(3, 1, 1)
plt.plot(df['Frame'], df['Front_Knee_Angle'], color='green', linewidth=2, label='Knee Angle')
plt.axhline(y=130, color='r', linestyle='--', alpha=0.5, label='Drive Threshold (<130)')
plt.title('Front Knee Flexion (Compression)')
plt.ylabel('Angle (Degrees)')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Backlift / Arm Extension
plt.subplot(3, 1, 2)
plt.plot(df['Frame'], df['Lead_Arm_Angle'], color='blue', linewidth=2, label='Lead Arm Extension')
plt.title('Lead Arm Extension (Backlift & Swing)')
plt.ylabel('Angle (Degrees)')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Head Stability (The Technical Metric)
plt.subplot(3, 1, 3)
plt.bar(df['Frame'], df['Head_Deviation_X'], color='orange', alpha=0.7, label='Head Deviation')
plt.axhline(y=10, color='red', linestyle='--', label='Unstable Threshold')
plt.title('Head Stability (X-Axis Deviation)')
plt.ylabel('Deviation (Pixels)')
plt.xlabel('Frame Number')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('outputs/biomechanics_dashboard.png')
print("Dashboard saved to outputs/biomechanics_dashboard.png")
plt.show()