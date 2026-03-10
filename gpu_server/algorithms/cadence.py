"""
Cadence (Adım Hızı - SPM) Analizi
=================================
Bu betik, sağ ve sol ayak temas (foot strike) anlarını kullanarak
genel (Total) ve zamana bağlı (Over Time) cadence değerlerini
"Dakikadaki Adım Sayısı (Steps Per Minute - SPM)" cinsinden hesaplar.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import json

def load_foot_strikes_from_jsonl(filepath):
    left_strikes = []
    right_strikes = []

    with open(filepath, "r") as f:
        for line in f:
            data = json.loads(line)

            step = {
                "time": data["time"],
                "pos": data["pos"]
            }

            if data["foot"] == "L":
                left_strikes.append(step)
            elif data["foot"] == "R":
                right_strikes.append(step)

    return left_strikes, right_strikes

def calculate_cadence(left_strikes, right_strikes):
    all_steps =[]
    for step in left_strikes:
        all_steps.append({'time': step['time'], 'foot': 'L', 'pos': step['pos']})
    for step in right_strikes:
        all_steps.append({'time': step['time'], 'foot': 'R', 'pos': step['pos']})

    all_steps = sorted(all_steps, key=lambda x: x['time'])

    if len(all_steps) < 2:
        raise ValueError("Cadence hesaplamak için en az 2 adım gereklidir.")

    total_steps = len(all_steps)
    first_time = all_steps[0]['time']
    last_time = all_steps[-1]['time']

    total_duration_sec = last_time - first_time
    total_duration_min = total_duration_sec / 60.0

    if total_duration_min <= 0:
        total_cadence = 0
    else:
        total_cadence = (total_steps - 1) / total_duration_min

    step_times = []
    instant_cadences =[]

    for i in range(1, len(all_steps)):
        t_prev = all_steps[i-1]['time']
        t_curr = all_steps[i]['time']

        delta_t = t_curr - t_prev
        delta_t = max(delta_t, 0.001)

        inst_cadence = 60.0 / delta_t

        step_times.append(t_curr)
        instant_cadences.append(inst_cadence)

    instant_cadences = np.array(instant_cadences)

    smoothed_cadences = uniform_filter1d(instant_cadences, size=3)

    return {
        'total_cadence': total_cadence,
        'step_times': step_times,
        'instant_cadences': instant_cadences,
        'smoothed_cadences': smoothed_cadences,
        'all_steps_sorted': all_steps
    }

def plot_cadence(left_strikes, right_strikes):
    res = calculate_cadence(left_strikes, right_strikes)

    t = res['step_times']
    inst_cad = res['instant_cadences']
    smooth_cad = res['smoothed_cadences']
    total_cad = res['total_cadence']

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(t, inst_cad, marker='o', linestyle='-', color='lightgray',
            markersize=5, linewidth=1.5, label='Anlık (Ham) Cadence')

    ax.plot(t, smooth_cad, marker='', linestyle='-', color='#8e44ad',
            linewidth=3, label='Düzeltilmiş (Smoothed) Cadence')

    ax.axhline(total_cad, color='#27ae60', linestyle='--', linewidth=2.5,
               label=f'Total Cadence: {total_cad:.1f} SPM')

    for step in res['all_steps_sorted']:
        color = '#3498db' if step['foot'] == 'L' else '#e74c3c'
        marker = 'L' if step['foot'] == 'L' else 'R'
        ax.text(step['time'], min(inst_cad) - 5, marker, color=color,
                fontsize=9, ha='center', fontweight='bold')

    ax.set_title("Cadence Over Time (Zamana Bağlı Adım Hızı)\n", fontsize=15, fontweight='bold')
    ax.set_xlabel("Zaman (saniye veya frame)\nL: Sol Ayak Teması, R: Sağ Ayak Teması", fontsize=12)
    ax.set_ylabel("Cadence (Adım / Dakika - SPM)", fontsize=12)

    y_min = min(min(inst_cad), total_cad) - 10
    y_max = max(max(inst_cad), total_cad) + 10
    ax.set_ylim(y_min, y_max)

    summary = f" ÖZET \n{'='*15}\n Toplam Adım: {len(res['all_steps_sorted'])}\n Ortalama Hız: {total_cad:.1f} SPM"
    ax.text(0.02, 0.95, summary, transform=ax.transAxes, fontsize=11,
            va='top', bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.9))

    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    left_foot_strikes, right_foot_strikes = load_foot_strikes_from_jsonl("./nlf_outputs/foot_strikes.jsonl")

    print(f"Sol ayak adım sayısı: {len(left_foot_strikes)}")
    print(f"Sağ ayak adım sayısı: {len(right_foot_strikes)}")

    plot_cadence(left_foot_strikes, right_foot_strikes)