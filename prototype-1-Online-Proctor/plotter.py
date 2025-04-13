import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Load the data from CSV
df = pd.read_csv("final_suspicious_activity_data.csv")

# Create a time axis based on frame count and FPS (assuming 30 FPS)
start_time = datetime.now()
df['time'] = [start_time + timedelta(seconds=i/30) for i in range(len(df))]

# Create a figure with 4 subplots
fig, axes = plt.subplots(4, 1, figsize=(15, 20), sharex=True)
plt.suptitle('Behavior Monitoring Over Time', fontsize=16, y=1.02)

# Plot 1: Blink Count
axes[0].plot(df['time'], df['blink_total'], label='Blink Count', color='blue', marker='o', markersize=3)
axes[0].set_title('Blink Count', fontsize=14)  # Use set_title for subplot titles
axes[0].set_ylabel('Blink Count', fontsize=12)
axes[0].grid(True, linestyle='--', linewidth=0.5)
axes[0].legend(loc='upper left')

# Plot 2: Voice Detection (binary)
axes[1].plot(df['time'], df['voice_detected'], label='Voice Detected', color='red', linestyle='--', marker='x', markersize=5)
axes[1].set_title('Voice Detection', fontsize=14)
axes[1].set_ylabel('Voice Detected (0/1)', fontsize=12)
axes[1].set_yticks([0, 1])
axes[1].grid(True, linestyle='--', linewidth=0.5)
axes[1].legend(loc='upper left')

# Plot 3: Lip Opens
axes[2].plot(df['time'], df['lip_open_count'], label='Lip Opens', color='green', marker='s', markersize=3)
axes[2].set_title('Lip Opens', fontsize=14)
axes[2].set_ylabel('Lip Open Count', fontsize=12)
axes[2].grid(True, linestyle='--', linewidth=0.5)
axes[2].legend(loc='upper left')

# Plot 4: Gaze Direction (-1: Left, 0: Forward, 1: Right)
axes[3].plot(df['time'], df['gaze_direction'], label='Gaze Direction', color='purple', linestyle=':', marker='^', markersize=4)
axes[3].set_title('Gaze Direction', fontsize=14)
axes[3].set_ylabel('Gaze Direction', fontsize=12)
axes[3].set_yticks([-1, 0, 1])
axes[3].set_yticklabels(['Left', 'Forward', 'Right'])
axes[3].grid(True, linestyle='--', linewidth=0.5)
axes[3].legend(loc='upper left')

# Format x-axis to show time properly
axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
axes[3].xaxis.set_major_locator(mdates.SecondLocator(interval=10))
fig.autofmt_xdate()

# Adjust layout and save to PDF
plt.tight_layout()
plt.savefig('behavior_monitoring_report.pdf', format='pdf', bbox_inches='tight');
#plt.show()