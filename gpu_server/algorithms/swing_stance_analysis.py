"""
Ankle X Coordinate Analysis
===========================
X ekseni dünya referanslı - yürüyüş/koşu yönünde ilerlemeyi gösterir.
Swing phase: Ayak havadayken X koordinatı hızla değişir (öne doğru)
Stance phase: Ayak yerdeyken X koordinatı nispeten sabit kalır
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d, median_filter

L_ANKLE = 7
R_ANKLE = 8
L_FOOT = 10
R_FOOT = 11
PELVIS = 0


def load_nlf(filepath):
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
    
    spike_threshold: toplam sürenin bu oranından kısa olan non-zero bölgeler spike sayılır
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


def find_gait_cycles(velocity_signal):
    """
    Gait cycle'ları bul.
    
    Bir gait cycle: velocity'nin 0'a düştüğü andan (stance başlangıcı)
    tekrar 0'a düşeceği ana kadar geçen süre.
    
    Returns: List of (start_idx, end_idx) tuples
    """
    is_stance = velocity_signal == 0
    cycles = []
    
    # 0'a geçiş noktalarını bul (stance başlangıçları)
    stance_starts = []
    for i in range(1, len(is_stance)):
        # non-zero'dan 0'a geçiş = stance başlangıcı
        if not is_stance[i-1] and is_stance[i]:
            stance_starts.append(i)
    
    # Her iki ardışık stance başlangıcı arasında bir gait cycle var
    for i in range(len(stance_starts) - 1):
        start = stance_starts[i]
        end = stance_starts[i + 1]
        cycles.append((start, end))
    
    return cycles


def calculate_flight_metrics(left_vx, right_vx, timestamps):
    """
    Gait cycle bazında havada kalma yüzdelerini hesapla.
    
    Returns:
        dict: Her bir cycle ve ortalama değerler
    """
    # Timestamps for velocity (1 shorter than position)
    ts = timestamps[1:]
    
    # Sol ayak bazlı gait cycle'ları bul (ortadaki cycle'ları al)
    left_cycles = find_gait_cycles(left_vx)
    right_cycles = find_gait_cycles(right_vx)
    
    # Ortadaki cycle'ları seç (başlangıç ve bitiş kenar etkilerinden kaçın)
    def get_middle_cycles(cycles):
        if len(cycles) <= 2:
            return cycles
        # İlk ve son cycle'ı atla
        return cycles[1:-1]
    
    left_middle_cycles = get_middle_cycles(left_cycles)
    right_middle_cycles = get_middle_cycles(right_cycles)
    
    results = {
        'left_foot_cycles': [],
        'right_foot_cycles': [],
        'double_flight_cycles': [],
    }
    
    # Sol ayak cycle'ları için metrikleri hesapla
    for start, end in left_middle_cycles:
        cycle_len = end - start
        if cycle_len == 0:
            continue
        
        left_flight = np.sum(left_vx[start:end] != 0)
        right_flight = np.sum(right_vx[start:end] != 0)
        double_flight = np.sum((left_vx[start:end] != 0) & (right_vx[start:end] != 0))
        
        cycle_time = ts[end-1] - ts[start] if end-1 < len(ts) and start < len(ts) else cycle_len / 30.0
        
        results['left_foot_cycles'].append({
            'start_time': ts[start] if start < len(ts) else 0,
            'end_time': ts[end-1] if end-1 < len(ts) else 0,
            'cycle_frames': cycle_len,
            'left_flight_pct': (left_flight / cycle_len) * 100,
            'right_flight_pct': (right_flight / cycle_len) * 100,
            'double_flight_pct': (double_flight / cycle_len) * 100,
        })
    
    # Sağ ayak cycle'ları için metrikleri hesapla
    for start, end in right_middle_cycles:
        cycle_len = end - start
        if cycle_len == 0:
            continue
        
        left_flight = np.sum(left_vx[start:end] != 0)
        right_flight = np.sum(right_vx[start:end] != 0)
        double_flight = np.sum((left_vx[start:end] != 0) & (right_vx[start:end] != 0))
        
        results['right_foot_cycles'].append({
            'start_time': ts[start] if start < len(ts) else 0,
            'end_time': ts[end-1] if end-1 < len(ts) else 0,
            'cycle_frames': cycle_len,
            'left_flight_pct': (left_flight / cycle_len) * 100,
            'right_flight_pct': (right_flight / cycle_len) * 100,
            'double_flight_pct': (double_flight / cycle_len) * 100,
        })
    
    # Ortalama değerleri hesapla
    def calc_averages(cycle_list):
        if not cycle_list:
            return {'avg_left_flight': 0, 'avg_right_flight': 0, 'avg_double_flight': 0}
        return {
            'avg_left_flight': np.mean([c['left_flight_pct'] for c in cycle_list]),
            'avg_right_flight': np.mean([c['right_flight_pct'] for c in cycle_list]),
            'avg_double_flight': np.mean([c['double_flight_pct'] for c in cycle_list]),
        }
    
    results['left_cycle_averages'] = calc_averages(results['left_foot_cycles'])
    results['right_cycle_averages'] = calc_averages(results['right_foot_cycles'])
    
    # Genel ortalama (tüm cycle'ların ortalaması)
    all_cycles = results['left_foot_cycles'] + results['right_foot_cycles']
    results['overall_averages'] = calc_averages(all_cycles)
    
    return results


def plot_ankle_x(filepath):
    """Plot X coordinates over time for both ankles."""
    frames = load_nlf(filepath)
    fn = os.path.basename(filepath)
    
    timestamps = np.array([f['timestamp_sec'] for f in frames])
    left_ankle_x = np.array([f['joints_3d'][0][0][L_ANKLE][0] for f in frames])
    right_ankle_x = np.array([f['joints_3d'][0][0][R_ANKLE][0] for f in frames])
    
    # 1. İlk tur: Genel smoothing (noise azaltma)
    smooth_win = 7
    left_smooth = uniform_filter1d(left_ankle_x, size=smooth_win)
    right_smooth = uniform_filter1d(right_ankle_x, size=smooth_win)
    
    # 2. İkinci tur: Plateau smoothing (stance fazlarını düzleştir)
    window = 10
    std_thresh = 80  # Adjust based on data scale
    left_plateau = plateau_smooth(left_smooth, window=window, std_threshold=std_thresh)
    right_plateau = plateau_smooth(right_smooth, window=window, std_threshold=std_thresh)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Top: Raw → Smooth → Plateau
    ax1 = axes[0]
    ax1.plot(timestamps, left_ankle_x, label='Left (raw)', color='blue', alpha=0.2, lw=0.5)
    ax1.plot(timestamps, right_ankle_x, label='Right (raw)', color='red', alpha=0.2, lw=0.5)
    ax1.plot(timestamps, left_smooth, label='Left (smooth)', color='blue', alpha=0.4, lw=1)
    ax1.plot(timestamps, right_smooth, label='Right (smooth)', color='red', alpha=0.4, lw=1)
    ax1.plot(timestamps, left_plateau, label='Left (plateau)', color='blue', alpha=0.9, lw=1.5)
    ax1.plot(timestamps, right_plateau, label='Right (plateau)', color='red', alpha=0.9, lw=1.5)
    ax1.set_ylabel('X Coordinate (world)')
    ax1.set_title(f'Ankle X Coordinates (Smooth → Plateau)\n{fn}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom: Velocity from plateau-smoothed signal
    ax2 = axes[1]
    dt = np.diff(timestamps)
    dt[dt == 0] = 1e-6
    
    left_vx = np.diff(left_plateau) / dt
    right_vx = np.diff(right_plateau) / dt
    
    # 1. Grafik smoothlaştırma (velocity smoothing)
    left_vx_smooth = uniform_filter1d(left_vx, size=3)
    right_vx_smooth = uniform_filter1d(right_vx, size=3)
    
    # 2. Velocity threshold: < 2000 → 0
    vel_threshold = 1500
    left_vx_thresh = left_vx_smooth.copy()
    right_vx_thresh = right_vx_smooth.copy()
    left_vx_thresh[left_vx_thresh < vel_threshold] = 0
    right_vx_thresh[right_vx_thresh < vel_threshold] = 0
    
    # 3. Peak temizleme (short spikes)
    left_vx_clean = remove_spikes(left_vx_thresh)
    right_vx_clean = remove_spikes(right_vx_thresh)
    
    ax2.plot(timestamps[1:], left_vx_clean, label='Left Vx', color='blue', alpha=0.8, lw=1.5)
    ax2.plot(timestamps[1:], right_vx_clean, label='Right Vx', color='red', alpha=0.8, lw=1.5)
    ax2.axhline(0, color='black', ls='--', lw=0.5)
    ax2.set_xlabel('Time (sec)')
    ax2.set_ylabel('X Velocity')
    ax2.set_title('Velocity (Stance=0, Swing=non-zero)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Havada kalma metriklerini hesapla
    flight_metrics = calculate_flight_metrics(left_vx_clean, right_vx_clean, timestamps)
    
    # Sonuçları yazdır
    print("\n" + "="*60)
    print("GAIT CYCLE BAZINDA HAVADA KALMA ANALİZİ")
    print("="*60)
    
    print("\n--- Sol Ayak Bazlı Cycle'lar ---")
    for i, cycle in enumerate(flight_metrics['left_foot_cycles']):
        print(f"  Cycle {i+1}: {cycle['start_time']:.2f}s - {cycle['end_time']:.2f}s ({cycle['cycle_frames']} frame)")
        print(f"    Sol ayak havada: {cycle['left_flight_pct']:.1f}%")
        print(f"    Sağ ayak havada: {cycle['right_flight_pct']:.1f}%")
        print(f"    İki ayak birden havada: {cycle['double_flight_pct']:.1f}%")
    
    print("\n--- Sağ Ayak Bazlı Cycle'lar ---")
    for i, cycle in enumerate(flight_metrics['right_foot_cycles']):
        print(f"  Cycle {i+1}: {cycle['start_time']:.2f}s - {cycle['end_time']:.2f}s ({cycle['cycle_frames']} frame)")
        print(f"    Sol ayak havada: {cycle['left_flight_pct']:.1f}%")
        print(f"    Sağ ayak havada: {cycle['right_flight_pct']:.1f}%")
        print(f"    İki ayak birden havada: {cycle['double_flight_pct']:.1f}%")
    
    print("\n--- ORTALAMALAR ---")
    avgs = flight_metrics['overall_averages']
    print(f"  Sol ayak havada kalma (ort): {avgs['avg_left_flight']:.1f}%")
    print(f"  Sağ ayak havada kalma (ort): {avgs['avg_right_flight']:.1f}%")
    print(f"  İki ayak birden havada (ort): {avgs['avg_double_flight']:.1f}%")
    print("="*60 + "\n")
    
    return {
        'timestamps': timestamps,
        'left_ankle_x': left_ankle_x,
        'right_ankle_x': right_ankle_x,
        'left_plateau': left_plateau,
        'right_plateau': right_plateau,
        'left_vx': left_vx_clean,
        'right_vx': right_vx_clean,
        'flight_metrics': flight_metrics,
    }


def main():
    folder = "nlf_outputs"
    print("\n=== Ankle X Coordinate Analysis ===\n")
    
    files = sorted(f for f in os.listdir(folder) if f.endswith(".jsonl"))
    for i, f in enumerate(files):
        print(f"  {i}: {f}")
    
    idx = int(input("\nDosya no: "))
    plot_ankle_x(os.path.join(folder, files[idx]))


if __name__ == "__main__":
    main()
