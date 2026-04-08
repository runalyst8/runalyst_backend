"""
Knee Flexion Analysis
=====================
Diz fleksiyonu analizi için temel yapı.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

SMPL_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1",
    "left_knee", "right_knee", "spine2", "left_ankle",
    "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hand", "right_hand",
]
# Joint indices
L_ANKLE = 7
R_ANKLE = 8
L_KNEE = 4
R_KNEE = 5
L_HIP = 1
R_HIP = 2


def load_nlf(filepath):
    """NLF dosyasını yükle."""
    with open(filepath) as f:
        return [json.loads(line) for line in f]


def plateau_smooth(signal, window=10, std_threshold=50):
    """
    Plateau detection: Düşük varyans olan bölgeleri düzleştir.
    
    - window: kaç frame'lik pencereler incelenecek
    - std_threshold: bu eşiğin altındaki std'ye sahip pencereler "sabit" sayılır
    
    Stance fazında ayak sabit → düşük std → ortalamaya eşitle
    Swing fazında ayak hareket → yüksek std → olduğu gibi bırak
    """
    result = signal.copy()
    n = len(signal)
    
    i = 0
    while i < n:
        end = min(i + window, n)
        segment = signal[i:end]
        
        if len(segment) < 3:
            i = end
            continue
            
        std = np.std(segment)
        
        if std < std_threshold:
            # Low variance = stance phase, flatten to mean
            result[i:end] = np.mean(segment)
        
        i = end
    
    return result


def remove_spikes(signal, spike_threshold=0.3):
    """
    Kısa süreli spike'ları temizle.
    
    Velocity sinyalinde:
    - Eğer bir değişim çok kısa süreli ise (1-2 frame) ve etrafı sıfır ise → spike
    - Bu spike'ları sıfıra eşitle
    """
    result = signal.copy()
    n = len(signal)
    
    # Non-zero bölgeleri bul
    threshold = np.std(signal) * 0.1  # Çok küçük değerler sıfır sayılsın
    is_active = np.abs(signal) > threshold
    
    # Bölgeleri işaretle
    i = 0
    while i < n:
        if is_active[i]:
            # Non-zero bölgenin başlangıcı
            start = i
            while i < n and is_active[i]:
                i += 1
            end = i
            duration = end - start
            
            # Çok kısa süreli bölge = spike
            if duration <= 3:  # 3 frame veya daha az
                result[start:end] = 0
        else:
            i += 1
    
    return result


def calculate_angle(p1, p2, p3):
    """
    3 nokta arasındaki açıyı hesapla (p2 köşe noktası).
    
    p1: Hip
    p2: Knee (vertex)
    p3: Ankle
    
    Returns: Açı (derece cinsinden)
    """
    v1 = p1 - p2  # Knee → Hip vektörü
    v2 = p3 - p2  # Knee → Ankle vektörü
    
    # Dot product ve magnitudes
    dot = np.dot(v1, v2)
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    
    if mag1 == 0 or mag2 == 0:
        return 180.0
    
    cos_angle = np.clip(dot / (mag1 * mag2), -1.0, 1.0)
    angle = np.arccos(cos_angle)
    
    return np.degrees(angle)


def find_gait_events(velocity_signal, knee_angles):
    """
    Velocity sinyalinden gait event'lerini bul.
    
    Returns:
        dict with lists of:
        - foot_strike: (frame_idx, angle) - stance başlangıcı
        - mid_stance: (frame_idx, angle) - stance ortası
        - toe_off: (frame_idx, angle) - stance sonu
        - mid_swing: (frame_idx, angle) - swing'deki min açı
    """
    n = len(velocity_signal)
    is_stance = velocity_signal == 0
    
    events = {
        'foot_strike': [],
        'mid_stance': [],
        'toe_off': [],
        'mid_swing': [],
    }
    
    # Stance ve swing bölgelerini bul
    i = 0
    while i < n:
        # Stance bölgesi bul (velocity == 0)
        if is_stance[i]:
            stance_start = i
            while i < n and is_stance[i]:
                i += 1
            stance_end = i - 1
            
            # Stance bölgesi yeterince uzunsa
            if stance_end - stance_start >= 2:
                # Foot strike: stance başlangıcı
                fs_idx = stance_start
                events['foot_strike'].append((fs_idx, knee_angles[fs_idx]))
                
                # Mid-stance: stance fazındaki minimum açı
                stance_angles = knee_angles[stance_start:stance_end+1]
                min_idx_local = np.argmin(stance_angles)
                ms_idx = stance_start + min_idx_local
                events['mid_stance'].append((ms_idx, knee_angles[ms_idx]))
                
                # Toe off: stance sonu
                to_idx = stance_end
                events['toe_off'].append((to_idx, knee_angles[to_idx]))
        
        # Swing bölgesi bul (velocity != 0)
        elif not is_stance[i]:
            swing_start = i
            while i < n and not is_stance[i]:
                i += 1
            swing_end = i - 1
            
            # Swing bölgesi yeterince uzunsa
            if swing_end - swing_start >= 2:
                # Mid-swing: minimum açı
                swing_angles = knee_angles[swing_start:swing_end+1]
                min_idx_local = np.argmin(swing_angles)
                min_idx = swing_start + min_idx_local
                events['mid_swing'].append((min_idx, knee_angles[min_idx]))
        else:
            i += 1
    
    return events


def plot_ankle_x(filepath, save_path=None):
    """Ankle X velocity ve diz açısı grafiği."""
    frames = load_nlf(filepath)
    fn = os.path.basename(filepath)
    
    timestamps = np.array([f['timestamp_sec'] for f in frames])
    left_ankle_x = np.array([f['joints_3d'][0][0][L_ANKLE][0] for f in frames])
    right_ankle_x = np.array([f['joints_3d'][0][0][R_ANKLE][0] for f in frames])
    
    # Diz açısı hesapla
    left_knee_angles = []
    right_knee_angles = []
    for f in frames:
        joints = f['joints_3d'][0][0]
        
        # Sol diz açısı: left_hip - left_knee - left_ankle
        l_hip = np.array(joints[L_HIP])
        l_knee = np.array(joints[L_KNEE])
        l_ankle = np.array(joints[L_ANKLE])
        left_knee_angles.append(calculate_angle(l_hip, l_knee, l_ankle))
        
        # Sağ diz açısı: right_hip - right_knee - right_ankle
        r_hip = np.array(joints[R_HIP])
        r_knee = np.array(joints[R_KNEE])
        r_ankle = np.array(joints[R_ANKLE])
        right_knee_angles.append(calculate_angle(r_hip, r_knee, r_ankle))
    
    left_knee_angles = np.array(left_knee_angles)
    right_knee_angles = np.array(right_knee_angles)
    
    # Açıları smoothla (zigzag'ları gider)
    angle_smooth_win = 5
    left_knee_smooth = uniform_filter1d(left_knee_angles, size=angle_smooth_win)
    right_knee_smooth = uniform_filter1d(right_knee_angles, size=angle_smooth_win)
    
    # 1. İlk tur: Genel smoothing (noise azaltma)
    smooth_win = 7
    left_smooth = uniform_filter1d(left_ankle_x, size=smooth_win)
    right_smooth = uniform_filter1d(right_ankle_x, size=smooth_win)
    
    # 2. İkinci tur: Plateau smoothing (stance fazlarını düzleştir)
    window = 10
    std_thresh = 80
    left_plateau = plateau_smooth(left_smooth, window=window, std_threshold=std_thresh)
    right_plateau = plateau_smooth(right_smooth, window=window, std_threshold=std_thresh)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Velocity hesapla
    dt = np.diff(timestamps)
    dt[dt == 0] = 1e-6
    
    left_vx = np.diff(left_plateau) / dt
    right_vx = np.diff(right_plateau) / dt
    
    # 1. Velocity smoothing
    left_vx_smooth = uniform_filter1d(left_vx, size=3)
    right_vx_smooth = uniform_filter1d(right_vx, size=3)
    
    # 2. Velocity threshold: < 1500 → 0
    vel_threshold = 1500
    left_vx_thresh = left_vx_smooth.copy()
    right_vx_thresh = right_vx_smooth.copy()
    left_vx_thresh[left_vx_thresh < vel_threshold] = 0
    right_vx_thresh[right_vx_thresh < vel_threshold] = 0
    
    # 3. Peak temizleme (short spikes)
    left_vx_clean = remove_spikes(left_vx_thresh)
    right_vx_clean = remove_spikes(right_vx_thresh)
    
    # Gait events bul (smoothed angles kullan)
    # Not: velocity 1 frame kısa, bu yüzden angles'ı da 1: ile al
    left_events = find_gait_events(left_vx_clean, left_knee_smooth[1:])
    right_events = find_gait_events(right_vx_clean, right_knee_smooth[1:])
    
    # Üst grafik: Velocity
    ax1 = axes[0]
    ax1.plot(timestamps[1:], left_vx_clean, label='Left Vx', color='blue', alpha=0.8, lw=1.5)
    ax1.plot(timestamps[1:], right_vx_clean, label='Right Vx', color='red', alpha=0.8, lw=1.5)
    ax1.axhline(0, color='black', ls='--', lw=0.5)
    ax1.set_ylabel('X Velocity')
    ax1.set_title(f'Ankle X Velocity (Stance=0, Swing=non-zero)\n{fn}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Alt grafik: Diz açısı
    ax2 = axes[1]
    ax2.plot(timestamps, left_knee_smooth, label='Left Knee', color='blue', alpha=0.8, lw=1.5)
    ax2.plot(timestamps, right_knee_smooth, label='Right Knee', color='red', alpha=0.8, lw=1.5)
    
    # Event'leri işaretle (sol ayak - mavi tonları)
    for idx, angle in left_events['foot_strike']:
        ax2.plot(timestamps[idx+1], angle, 'o', color='darkblue', markersize=8)
    for idx, angle in left_events['mid_stance']:
        ax2.plot(timestamps[idx+1], angle, 's', color='blue', markersize=8)
    for idx, angle in left_events['toe_off']:
        ax2.plot(timestamps[idx+1], angle, '^', color='royalblue', markersize=8)
    for idx, angle in left_events['mid_swing']:
        ax2.plot(timestamps[idx+1], angle, 'D', color='deepskyblue', markersize=8)
    
    # Event'leri işaretle (sağ ayak - kırmızı tonları)
    for idx, angle in right_events['foot_strike']:
        ax2.plot(timestamps[idx+1], angle, 'o', color='darkred', markersize=8)
    for idx, angle in right_events['mid_stance']:
        ax2.plot(timestamps[idx+1], angle, 's', color='red', markersize=8)
    for idx, angle in right_events['toe_off']:
        ax2.plot(timestamps[idx+1], angle, '^', color='orangered', markersize=8)
    for idx, angle in right_events['mid_swing']:
        ax2.plot(timestamps[idx+1], angle, 'D', color='salmon', markersize=8)
    
    # Legend için dummy plots
    ax2.plot([], [], 'o', color='gray', markersize=8, label='Foot Strike')
    ax2.plot([], [], 's', color='gray', markersize=8, label='Mid Stance')
    ax2.plot([], [], '^', color='gray', markersize=8, label='Toe Off')
    ax2.plot([], [], 'D', color='gray', markersize=8, label='Mid Swing')
    
    ax2.set_xlabel('Time (sec)')
    ax2.set_ylabel('Knee Angle (degrees)')
    ax2.set_title('Knee Flexion Angle with Gait Events')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    return {
        'timestamps': timestamps,
        'left_ankle_x': left_ankle_x,
        'right_ankle_x': right_ankle_x,
        'left_plateau': left_plateau,
        'right_plateau': right_plateau,
        'left_vx': left_vx_clean,
        'right_vx': right_vx_clean,
        'left_knee_angles': left_knee_smooth,
        'right_knee_angles': right_knee_smooth,
        'left_events': left_events,
        'right_events': right_events,
    }


def main():
    folder = "nlf_outputs"
    print("\n=== Knee Flexion Analysis ===\n")
    
    files = sorted(f for f in os.listdir(folder) if f.endswith(".jsonl"))
    for i, f in enumerate(files):
        print(f"  {i}: {f}")
    
    idx = int(input("\nDosya no: "))
    plot_ankle_x(os.path.join(folder, files[idx]))


if __name__ == "__main__":
    main()
