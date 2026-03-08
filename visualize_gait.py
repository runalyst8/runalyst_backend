import matplotlib.pyplot as plt
from gait_segmentation import GaitAnalyzer
from io_utility import extract_ys_and_times, open_nlf_output_json
from math_utility import get_dt, smooth

def visualize(jsonl_path, side="right"):
    analyzer = GaitAnalyzer(jsonl_path)
    cycles = analyzer.analyze(side=side)
    
    # Extract raw data for plotting
    video_nlf = open_nlf_output_json(jsonl_path)
    times, ys = extract_ys_and_times(video_nlf, "nlf.joints3d[0]", f"{side}_heel")
    dts = get_dt(times)
    ys_smooth = smooth(ys, dts)
    
    plt.figure(figsize=(12, 6))
    plt.plot(times, ys_smooth, label=f'{side.capitalize()} Heel (Smoothed)', color='blue')
    
    # Plot cycles
    for i, c in enumerate(cycles):
        # Stance phase (green background)
        plt.axvspan(c.start_time, c.toe_off_time, color='green', alpha=0.2, label='Stance' if i==0 else "")
        # Swing phase (red background)
        plt.axvspan(c.toe_off_time, c.end_time, color='red', alpha=0.1, label='Swing' if i==0 else "")
        
        # Mark events
        plt.axvline(c.start_time, color='g', linestyle='--', alpha=0.5)
        plt.axvline(c.toe_off_time, color='r', linestyle='--', alpha=0.5)
        
    plt.title(f'Gait Segmentation Analysis - {side.capitalize()} Side')
    plt.xlabel('Time (s)')
    plt.ylabel('Vertical Position (Normalized)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'gait_analysis_{side}.png')
    print(f"Saved gait_analysis_{side}.png")

if __name__ == "__main__":
    visualize("run_nlf.jsonl", "right")
