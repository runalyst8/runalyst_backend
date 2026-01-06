import numpy as np
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from io_utility import open_nlf_output_json, extract_ys_and_times
from math_utility import get_dt, smooth, find_y_peaks

@dataclass
class GaitEvent:
    type: str  # "FS" (Foot Strike) or "TO" (Toe Off)
    time: float
    index: int

@dataclass
class GaitCycle:
    start_time: float  # FS
    end_time: float    # Next FS
    toe_off_time: float # TO
    stance_duration: float
    swing_duration: float
    side: str

class GaitAnalyzer:
    def __init__(self, jsonl_path: str):
        self.jsonl_path = jsonl_path
        self.video_nlf = open_nlf_output_json(jsonl_path)
        
    def analyze(self, side: str = "right") -> List[GaitCycle]:
        """
        Analyze gait for the specified side (left/right).
        Returns a list of GaitCycle objects.
        """
        # 1. Component Selection (Heel is usually more robust for initial contact detection)
        # However, for Stance Phase boundaries, we often define:
        # FS = Heel Strike
        # TO = Toe Off
        # So we might need both signals or a composite.
        # For this prototype, we'll try to use the 'Heel' signal for FS and 'Toe' for TO if possible,
        # but let's stick to the single robust signal approach first if 'analyze_foot_strike' suggests Y peaks are contact.
        
        # Taking 'heel' as the primary driver for Stance Phase logic similar to analyze_foot_strike.py
        target_path = "nlf.joints3d[0]" # standard
        
        # Extract Heel Data
        times, ys = extract_ys_and_times(self.video_nlf, target_path, f"{side}_heel")
        dts = get_dt(times)
        ys_smooth = smooth(ys, dts)
        
        # Find Peaks (Mid-Stance approximations according to analyze_foot_strike detection logic)
        peak_indices = find_y_peaks(ys_smooth, dts)
        
        cycles: List[GaitCycle] = []
        
        # 2. Refine timestamps for each peak (Stance Phase)
        # We assume each Peak represents a stance phase.
        # We need to find the "Rise" (FS) and "Fall" (TO) around this peak.
        
        # Dynamic Threshold: Median + Fraction of StdDev
        # This helps isolate the "contact bump" from noise
        baseline = np.median(ys_smooth)
        std_dev = np.std(ys_smooth)
        threshold = baseline + 0.2 * std_dev
        
        events = []
        
        for p_idx in peak_indices:
            # Search backwards for FS (Crossing threshold upwards)
            fs_idx = p_idx
            while fs_idx > 0 and ys_smooth[fs_idx] > threshold:
                fs_idx -= 1
            # Refine FS: Local minimum before the rise? Or just the crossing?
            # Let's use the crossing point for stability in prototype.
            
            # Search forwards for TO (Crossing threshold downwards)
            to_idx = p_idx
            while to_idx < len(ys_smooth) - 1 and ys_smooth[to_idx] > threshold:
                to_idx += 1
                
            # Basic validation
            if fs_idx < p_idx < to_idx:
                fs_t = float(times[fs_idx])
                to_t = float(times[to_idx])
                
                # Check duration constraints (e.g., stance phase shouldn't be 0.01s or 2.0s for running)
                duration = to_t - fs_t
                if 0.1 <= duration <= 1.0: # Reasonable range for walking/running
                    events.append({"fs": fs_t, "to": to_t, "fs_idx": fs_idx, "to_idx": to_idx})

        # 3. Construct Cycles
        # Cycle N: FS_N to FS_{N+1}
        for i in range(len(events) - 1):
            current_stance = events[i]
            next_stance = events[i+1]
            
            fs_current = current_stance["fs"]
            to_current = current_stance["to"]
            fs_next = next_stance["fs"]
            
            stance_dur = to_current - fs_current
            swing_dur = fs_next - to_current
            
            # Sanity check for swing phase
            if swing_dur > 0:
                cycle = GaitCycle(
                    start_time=fs_current,
                    end_time=fs_next,
                    toe_off_time=to_current,
                    stance_duration=stance_dur,
                    swing_duration=swing_dur,
                    side=side
                )
                cycles.append(cycle)
                
        return cycles

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to .jsonl file")
    parser.add_argument("--side", choices=["left", "right"], default="right")
    args = parser.parse_args()
    
    analyzer = GaitAnalyzer(args.file)
    cycles = analyzer.analyze(side=args.side)
    
    print(f"Analyzed {args.file} ({args.side})")
    print(f"Detected {len(cycles)} cycles:")
    print("-" * 60)
    print(f"{'Cycle #':<10} {'Start (s)':<12} {'Toe-Off (s)':<12} {'End (s)':<12} {'Stance (s)':<12} {'Swing (s)':<12}")
    print("-" * 60)
    
    for i, c in enumerate(cycles):
        print(f"{i+1:<10} {c.start_time:<12.3f} {c.toe_off_time:<12.3f} {c.end_time:<12.3f} {c.stance_duration:<12.3f} {c.swing_duration:<12.3f}")

if __name__ == "__main__":
    main()
