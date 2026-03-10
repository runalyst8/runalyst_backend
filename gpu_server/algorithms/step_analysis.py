"""
Foot Progression Angle (İçe/Dışa Basma) Analizi (v1)
===================================================

Metrik: Ayak İlerleme Açısı (Foot Progression Angle - FPA)
----------------------------------------------------------
  1. Yürüyüş Yönü (Global Forward): Tüm framelerdeki sol ve sağ
     (ayak_ucu - bilek) vektörlerinin medyanı
     Kamera açısından veya yürüme bandından bağımsız çalışır
  2. FPA: Yürüyüş yönü ile topuk-parmak ucu vektörü arasındaki açı
     Sadece ayak YERE BASTIĞINDA (stance phase) ölçülür

Yön Kuralı (Sağ ve Sol ayak için simetrik):
  • Pozitif (+) Açı : Dışa dönük (Out-toeing / Duck feet)
  • Negatif (-) Açı : İçe dönük (Pigeon toes)

Klinik Eşikler (Literatür Ortalamaları)
---------------------------------------
   FPA < 0°          → İçe Basma (In-toeing)
   0° ≤ FPA ≤ 15°    → Normal (Hafif dışa dönüklük anatomik olarak normal)
   FPA > 15°         → Dışa Basma (Out-toeing)

Sınırlamalar
------------
  • Sadece düz yürüyüşler (treadmill veya koridor)
  • Dönüş içeren videolarda "Global Forward" vektörü sapabilir
"""


import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

# ── SMPL Joint Tanımları ──────────────────────────────────────────
L_ANKLE = 7;
R_ANKLE = 8
L_FOOT = 10;
R_FOOT = 11
PELVIS = 0

# ── Eşikler (Derece) ─────────────────────────────────────────────
TH_INWARD = 0.0
TH_OUTWARD = 15.0

def load_nlf(fp):
    with open(fp) as f:
        return [json.loads(l) for l in f]


def get_j3d(frames, idx):
    return np.array([f["joints_3d"][0][0][idx] for f in frames])


def detect_contacts(y_sig, min_dist=10, smooth_win=5, prom_ratio=0.15):
    sm = uniform_filter1d(y_sig.astype(float), size=smooth_win)
    yr = sm.max() - sm.min()
    peaks, _ = find_peaks(sm, distance=min_dist, prominence=yr * prom_ratio)
    return peaks


def classify_fpa(angle):
    if angle < TH_INWARD:
        return "İçe Basma (In-toeing)"
    elif angle > TH_OUTWARD:
        return "Dışa Basma (Out-toeing)"
    else:
        return "Normal"


def normalize_vec(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else v


def calculate_global_forward(l_ankle_xz, l_foot_xz, r_ankle_xz, r_foot_xz):
    """XZ düzleminde çalışır"""
    l_vecs = l_foot_xz - l_ankle_xz
    r_vecs = r_foot_xz - r_ankle_xz

    all_vecs = np.vstack((l_vecs, r_vecs))
    norms = np.linalg.norm(all_vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    norm_vecs = all_vecs / norms

    global_fwd = np.median(norm_vecs, axis=0)
    return normalize_vec(global_fwd)


def compute_fpa(ankle_xz, foot_xz, fwd_vec, is_right):
    foot_vec = foot_xz - ankle_xz

    if is_right:
        side_vec = np.array([fwd_vec[1], -fwd_vec[0]])
    else:
        side_vec = np.array([-fwd_vec[1], fwd_vec[0]])

    angles = []
    for v in foot_vec:
        v_norm = normalize_vec(v)
        dot_fwd = np.dot(v_norm, fwd_vec)
        dot_side = np.dot(v_norm, side_vec)

        angle = np.degrees(np.arctan2(dot_side, dot_fwd))
        angles.append(angle)

    return np.array(angles)


def analyze_progression(filepath):
    frames = load_nlf(filepath)

    l_ankle = get_j3d(frames, L_ANKLE);
    r_ankle = get_j3d(frames, R_ANKLE)
    l_foot = get_j3d(frames, L_FOOT);
    r_foot = get_j3d(frames, R_FOOT)
    pelvis = get_j3d(frames, PELVIS)

    l_ankle_xz = l_ankle[:, [0, 2]];
    l_foot_xz = l_foot[:, [0, 2]]
    r_ankle_xz = r_ankle[:, [0, 2]];
    r_foot_xz = r_foot[:, [0, 2]]

    l_contacts = detect_contacts(np.maximum(l_ankle[:, 1], l_foot[:, 1]))
    r_contacts = detect_contacts(np.maximum(r_ankle[:, 1], r_foot[:, 1]))

    fwd_vec = calculate_global_forward(l_ankle_xz, l_foot_xz, r_ankle_xz, r_foot_xz)

    l_angles = compute_fpa(l_ankle_xz, l_foot_xz, fwd_vec, is_right=False)
    r_angles = compute_fpa(r_ankle_xz, r_foot_xz, fwd_vec, is_right=True)

    l_fpa_contacts = l_angles[l_contacts] if len(l_contacts) > 0 else np.array([0])
    r_fpa_contacts = r_angles[r_contacts] if len(r_contacts) > 0 else np.array([0])

    avg_l_fpa = float(np.mean(l_fpa_contacts))
    avg_r_fpa = float(np.mean(r_fpa_contacts))

    return {
        "fwd_vec": fwd_vec,
        "l_angles": l_angles, "r_angles": r_angles,
        "l_contacts": l_contacts, "r_contacts": r_contacts,
        "avg_l_fpa": avg_l_fpa, "avg_r_fpa": avg_r_fpa,
        "l_class": classify_fpa(avg_l_fpa),
        "r_class": classify_fpa(avg_r_fpa),
        "pelvis_xz": pelvis[:, [0, 2]],
        "l_ankle_xz": l_ankle_xz, "l_foot_xz": l_foot_xz,
        "r_ankle_xz": r_ankle_xz, "r_foot_xz": r_foot_xz,
        "n_frames": len(frames)
    }


def plot_progression(filepath):
    r = analyze_progression(filepath)
    fn = os.path.basename(filepath)
    t = np.arange(r["n_frames"])

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"İçe/Dışa Basma (Foot Progression) Analizi: {fn}", fontsize=14, fontweight="bold")

    plots_cfg = [
        ("Sol Ayak FPA", r["l_angles"], r["l_contacts"], r["avg_l_fpa"], r["l_class"], "#3498db", axes[0, 0]),
        ("Sağ Ayak FPA", r["r_angles"], r["r_contacts"], r["avg_r_fpa"], r["r_class"], "#e74c3c", axes[0, 1])
    ]

    for title, angles, contacts, avg_fpa, cls, color, ax in plots_cfg:
        ax.plot(t, angles, color=color, alpha=0.5, label="FPA (Tüm zamanlar)")
        ax.plot(contacts, angles[contacts], "o", color=color, markersize=8, markeredgecolor='black',
                label="Temas Anları (Mid-Stance)")

        ax.axhline(0, color="black", ls="--", lw=1)
        ax.axhline(TH_OUTWARD, color="gray", ls=":", lw=1)
        ax.axhline(avg_fpa, color="green", ls="-", lw=2, label=f"Ortalama: {avg_fpa:+.1f}°")

        ax.fill_between(t, TH_INWARD, -45, alpha=0.1, color="purple", label="İçe Basma Alanı")
        ax.fill_between(t, TH_INWARD, TH_OUTWARD, alpha=0.1, color="green", label="Normal Alan")
        ax.fill_between(t, TH_OUTWARD, 45, alpha=0.1, color="orange", label="Dışa Basma Alanı")

        ax.set_ylim(-30, 45)
        ax.set_title(f"{title} - Sonuç: {cls}")
        ax.set_ylabel("Açı (Derece)\n(+) Dışa, (-) İçe")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(r["pelvis_xz"][:, 0], r["pelvis_xz"][:, 1], color="gray", ls="--", alpha=0.5, label="Pelvis Yolu")

    def plot_foot_arrows(contacts, ankle_xz, foot_xz, color, label):
        if len(contacts) == 0: return

        ax_pts = ankle_xz[contacts];
        fx_pts = foot_xz[contacts]
        ax.quiver(ax_pts[:, 0], ax_pts[:, 1], fx_pts[:, 0] - ax_pts[:, 0], fx_pts[:, 1] - ax_pts[:, 1],
                  angles='xy', scale_units='xy', scale=1, color=color, width=0.005, alpha=0.8, label=label)

    plot_foot_arrows(r["l_contacts"], r["l_ankle_xz"], r["l_foot_xz"], "#2980b9", "Sol Adımlar")
    plot_foot_arrows(r["r_contacts"], r["r_ankle_xz"], r["r_foot_xz"], "#c0392b", "Sağ Adımlar")

    mid_x = np.mean(r["pelvis_xz"][:, 0]);
    mid_z = np.mean(r["pelvis_xz"][:, 1])
    ax.quiver(mid_x, mid_z, r["fwd_vec"][0] * 0.2, r["fwd_vec"][1] * 0.2,
              angles='xy', scale_units='xy', scale=1, color="black", width=0.01, label="Genel İlerleme Yönü")

    ax.set_title("Kuşbakışı (Top-Down) Ayak Vektörleri")
    ax.set_xlabel("X (Yanal)");
    ax.set_ylabel("Z (İleri/Geri)")
    ax.axis("equal")
    ax.legend(fontsize=8);
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.axis("off")
    summary = (
        f"{'═' * 45}\n"
        f"  İÇE/DIŞA BASMA (FPA) SONUÇLARI\n"
        f"{'═' * 45}\n\n"
        f"  Dosya : {fn}\n"
        f"  Frame : {r['n_frames']}\n\n"
        f"  Sol Ayak (Değerlendirilen Adım: {len(r['l_contacts'])}):\n"
        f"    Ortalama FPA : {r['avg_l_fpa']:+.1f}°\n"
        f"    Durum        : {r['l_class']}\n\n"
        f"  Sağ Ayak (Değerlendirilen Adım: {len(r['r_contacts'])}):\n"
        f"    Ortalama FPA : {r['avg_r_fpa']:+.1f}°\n"
        f"    Durum        : {r['r_class']}\n\n"
        f"{'─' * 45}\n"
        f"  Değerlendirme Kriterleri:\n"
        f"    < 0°           : İçe Basma (Pigeon-toe)\n"
        f"    0° ile +15°    : Normal\n"
        f"    > +15°         : Dışa Basma (Duck-foot)\n"
        f"  *(Hesaplamalar ayak yerdeyken yapılmıştır)*\n"
        f"{'═' * 45}"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=11,
            va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="whitesmoke", alpha=0.9))

    plt.tight_layout()
    plt.show()


def batch_analyze(folder="nlf_outputs"):
    files = sorted(f for f in os.listdir(folder) if f.endswith(".jsonl"))

    print(f"\n{'═' * 85}")
    print("  TOPLU İÇE/DIŞA BASMA (FPA) ANALİZİ")
    print(f"{'═' * 85}\n")
    hdr = f"  {'Dosya':<35} | {'Sol FPA':>8} {'Sol Durum':<15} | {'Sağ FPA':>8} {'Sağ Durum':<15}"
    print(hdr)
    print(f"  {'-' * 85}")

    for f in files:
        r = analyze_progression(os.path.join(folder, f))
        print(f"  {f:<35} | {r['avg_l_fpa']:>+7.1f}° {r['l_class']:<15} | {r['avg_r_fpa']:>+7.1f}° {r['r_class']:<15}")

    print(f"  {'-' * 85}\n")


# ── CLI ───────────────────────────────────────────────────────────
def main():
    folder = "nlf_outputs"
    if not os.path.exists(folder):
        print(f"Hata: '{folder}' klasörü bulunamadı.")
        return

    print("\n=== İçe / Dışa Basma (Foot Progression) Analizi ===\n")
    print("  1: Tek dosya analizi (Kuşbakışı grafik ve açılar)")
    print("  2: Toplu analiz (Tüm dosyalar)")

    choice = input("\nSeçim (1/2): ").strip()

    if choice == "2":
        batch_analyze(folder)
    else:
        files = sorted(f for f in os.listdir(folder) if f.endswith(".jsonl"))
        for i, f in enumerate(files):
            print(f"  {i}: {f}")
        try:
            idx = int(input("\nDosya no: "))
            plot_progression(os.path.join(folder, files[idx]))
        except (ValueError, IndexError):
            print("Geçersiz seçim.")


if __name__ == "__main__":
    main()