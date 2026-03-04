"""
Heel / Mid / Toe Strike Analizi  (v4)
======================================

Metrik
------
  mn_max = min( max(L_norm_diff), max(R_norm_diff) )

  Burada:
    norm_diff  = (ankle_y − foot_y) / shin_length     [birim-siz]
    shin_length = median 3D mesafe(diz, ayak bileği)   [normalizasyon]

  Her taraftaki (sol/sağ) tüm frameler boyunca normalize farkın
  maksimumu alınır, sonra iki tarafın MİNİMUMU seçilir. Bu sayede:
    • Tek taraflı gürültü / outlier bastırılır
    • Her iki ayak da aynı eğilimi göstermeli → güvenilirlik artar
    • Ölçekten bağımsız (shin ile normalize)

Eşikler  (32 dosya üzerinden %90.6 doğrulukla kalibre)
-------
    mn_max > −0.038  →  Heel Strike
   −0.096 < mn_max ≤ −0.038  →  Mid Strike
    mn_max ≤ −0.096  →  Toe Strike

Doğrulama metriği
-----------------
  mn_amax = min( max(L_angle), max(R_angle) )   [derece]
    angle = atan2(ankle_y − foot_y, horizontal_dist)

    mn_amax > −6.5°  →  Heel
   −16.3° < mn_amax ≤ −6.5°  →  Mid
    mn_amax ≤ −16.3°  →  Toe

Sınırlamalar
------------
  • SMPL ankle ≠ topuk, foot ≠ parmak ucu
  • Her iki metrik de aynı 3 dosyada hata yapıyor (overstride toe
    etiketli ama biyomekanik olarak heel-benzeri)
  • 32 dosyalık veri setinden kalibrasyon → yeni kamera açılarında
    yeniden doğrulama önerilir
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d


# ── SMPL Joint Tanımları ──────────────────────────────────────────
SMPL_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1",
    "left_knee", "right_knee", "spine2", "left_ankle",
    "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hand", "right_hand",
]

L_KNEE = 4;  R_KNEE = 5
L_ANKLE = 7; R_ANKLE = 8
L_FOOT = 10; R_FOOT = 11


# ── Eşikler ──────────────────────────────────────────────────────
# Birincil: mn_max (normalize, birim-siz)
TH_HEEL = -0.038
TH_TOE  = -0.096

# Doğrulama: mn_amax (derece)
TH_HEEL_A = -6.5
TH_TOE_A  = -16.3


# ── Yardımcılar ──────────────────────────────────────────────────
def load_nlf(fp):
    with open(fp) as f:
        return [json.loads(l) for l in f]


def get_j3d(frames, idx):
    return np.array([f["joints_3d"][0][0][idx] for f in frames])


def detect_contacts(y_sig, min_dist=8, smooth_win=5, prom_ratio=0.15):
    sm = uniform_filter1d(y_sig.astype(float), size=smooth_win)
    yr = sm.max() - sm.min()
    peaks, _ = find_peaks(sm, distance=min_dist, prominence=yr * prom_ratio)
    return peaks, sm


def estimate_ground(y_arrays):
    return float(np.percentile(np.concatenate(y_arrays), 95))


def classify(mn_max):
    if mn_max > TH_HEEL:
        return "Heel Strike"
    elif mn_max > TH_TOE:
        return "Mid Strike"
    else:
        return "Toe Strike"


def classify_angle(mn_amax):
    if mn_amax > TH_HEEL_A:
        return "Heel Strike"
    elif mn_amax > TH_TOE_A:
        return "Mid Strike"
    else:
        return "Toe Strike"


def confidence(mn_max, mn_amax):
    """İki metrik uyuşuyorsa güven artar; eşik sınırındaysa düşer."""
    primary = classify(mn_max)
    secondary = classify_angle(mn_amax)

    # Eşik sınırına mesafe
    d_heel = abs(mn_max - TH_HEEL)
    d_toe  = abs(mn_max - TH_TOE)
    margin = min(d_heel, d_toe)

    if primary == secondary and margin > 0.025:
        return "yüksek"
    elif primary == secondary:
        return "orta"
    elif margin > 0.02:
        return "orta"
    else:
        return "düşük"


# ── Tek taraf analizi ─────────────────────────────────────────────
def analyze_side(frames, knee_i, ankle_i, foot_i):
    knee  = get_j3d(frames, knee_i)
    ankle = get_j3d(frames, ankle_i)
    foot  = get_j3d(frames, foot_i)

    shin_len = float(np.median(np.sqrt(np.sum((knee - ankle)**2, axis=1))))

    diff = ankle[:, 1] - foot[:, 1]
    norm_diff = diff / shin_len

    d = ankle - foot
    dxz = np.clip(np.sqrt(d[:, 0]**2 + d[:, 2]**2), 1e-6, None)
    angles = np.degrees(np.arctan2(d[:, 1], dxz))

    foot_prox = np.maximum(ankle[:, 1], foot[:, 1])
    contacts, smoothed = detect_contacts(foot_prox)
    if len(contacts) < 2:
        contacts = np.where(foot_prox >= np.percentile(foot_prox, 80))[0]

    return {
        "ankle_y": ankle[:, 1],
        "foot_y": foot[:, 1],
        "norm_diff": norm_diff,
        "angles": angles,
        "shin_len": shin_len,
        "max_nd": float(np.max(norm_diff)),
        "p95_nd": float(np.percentile(norm_diff, 95)),
        "max_angle": float(np.max(angles)),
        "contacts": contacts,
        "smoothed": smoothed,
    }


# ── Tam analiz ────────────────────────────────────────────────────
def analyze_strike(filepath):
    frames = load_nlf(filepath)

    left  = analyze_side(frames, L_KNEE, L_ANKLE, L_FOOT)
    right = analyze_side(frames, R_KNEE, R_ANKLE, R_FOOT)

    # Birincil metrik: mn_max
    mn_max = min(left["max_nd"], right["max_nd"])
    # Doğrulama: mn_amax
    mn_amax = min(left["max_angle"], right["max_angle"])

    strike = classify(mn_max)
    strike_a = classify_angle(mn_amax)
    conf = confidence(mn_max, mn_amax)

    ground = estimate_ground([left["ankle_y"], right["ankle_y"],
                              left["foot_y"], right["foot_y"]])

    return {
        "left": left, "right": right,
        "mn_max": mn_max, "mn_amax": mn_amax,
        "type": strike, "type_angle": strike_a,
        "confidence": conf,
        "ground": ground,
        "n_frames": len(frames),
    }


# ── Görselleştirme ───────────────────────────────────────────────
def plot_strike(filepath):
    r = analyze_strike(filepath)
    fn = os.path.basename(filepath)
    n = r["n_frames"]
    t = np.arange(n)

    fig, axes = plt.subplots(3, 2, figsize=(16, 13))
    fig.suptitle(
        f"Strike Analizi: {fn}\n"
        f"Sonuç: {r['type']}  ({r['confidence']} güven)   "
        f"[mn_max={r['mn_max']:+.4f},  mn_amax={r['mn_amax']:+.1f}°]",
        fontsize=13, fontweight="bold",
    )

    sides_cfg = {"left": ("Sol", "#e74c3c", "#3498db"),
                 "right": ("Sağ", "#c0392b", "#2980b9")}

    for col, (key, (tag, ca, cf)) in enumerate(sides_cfg.items()):
        s = r[key]

        # ── Üst: Y koordinatları ──
        ax = axes[0, col]
        ax.plot(t, -s["ankle_y"], label=f"{key}_ankle", lw=1.3, color=ca)
        ax.plot(t, -s["foot_y"],  label=f"{key}_foot",  lw=1.3, color=cf)
        ax.axhline(-r["ground"], color="brown", ls=":", lw=1, alpha=.6,
                    label="zemin")
        for p in s["contacts"]:
            ax.axvline(p, color="green", alpha=0.2, lw=0.7)
        ax.set_title(f"{tag} Ayak – Y (ters)  [yeşil = temas]")
        ax.set_ylabel("-Y (yukarı = yere yakın)")
        ax.legend(fontsize=8); ax.grid(True, alpha=.25)

        # ── Orta: Normalize fark ──
        ax = axes[1, col]
        ax.plot(t, s["norm_diff"], lw=1.1, color="#8e44ad", alpha=.7,
                label="norm_diff")
        ax.axhline(s["max_nd"], color="red", ls="-", lw=1.5, alpha=.7,
                    label=f"max = {s['max_nd']:+.4f}")
        ax.axhline(TH_HEEL, color="#e67e22", ls="--", lw=1, alpha=.5,
                    label=f"Heel/Mid ({TH_HEEL})")
        ax.axhline(TH_TOE, color="#e74c3c", ls="--", lw=1, alpha=.5,
                    label=f"Mid/Toe ({TH_TOE})")
        ax.axhline(0, color="black", ls=":", lw=.8, alpha=.3)
        ax.fill_between(t, s["norm_diff"], TH_HEEL,
                        where=s["norm_diff"] > TH_HEEL, alpha=.15, color="#e74c3c")
        ax.fill_between(t, s["norm_diff"], TH_TOE,
                        where=s["norm_diff"] < TH_TOE, alpha=.15, color="#3498db")
        ax.set_title(
            f"{tag}: (ankle_y−foot_y)/shin  |  max={s['max_nd']:+.4f} → {classify(s['max_nd'])}"
        )
        ax.set_ylabel("Norm. fark"); ax.set_xlabel("Frame")
        ax.legend(fontsize=7); ax.grid(True, alpha=.25)

    # ── Alt sol: Her iki tarafın max karşılaştırması ──
    ax = axes[2, 0]
    for key, clr, tag in [("left", "#9b59b6", "Sol"), ("right", "#f39c12", "Sağ")]:
        ax.hist(r[key]["norm_diff"], bins=30, alpha=.45, color=clr, label=tag)
    ax.axvline(TH_HEEL, color="#e67e22", ls="--", lw=1.5, label=f"Heel/Mid ({TH_HEEL})")
    ax.axvline(TH_TOE,  color="#e74c3c", ls="--", lw=1.5, label=f"Mid/Toe ({TH_TOE})")
    ax.axvline(0, color="black", ls=":", lw=.8, alpha=.4)

    # mn_max çizgisi
    ax.axvline(r["mn_max"], color="green", ls="-", lw=2, alpha=.8,
               label=f"mn_max = {r['mn_max']:+.4f}")
    ax.set_title("Normalize fark dağılımı + mn_max")
    ax.set_xlabel("(ankle_y − foot_y) / shin"); ax.set_ylabel("Frame")
    ax.legend(fontsize=7); ax.grid(True, alpha=.25)

    # ── Alt sağ: Sonuç özeti ──
    ax = axes[2, 1]
    ax.axis("off")
    ls, rs = r["left"], r["right"]
    agree = "✓ UYUMLU" if r["type"] == r["type_angle"] else "✗ UYUMSUZ"
    summary = (
        f"{'═'*46}\n"
        f"  STRIKE ANALİZ SONUÇLARI (v4)\n"
        f"{'═'*46}\n\n"
        f"  Dosya : {fn}\n"
        f"  Frame : {n}\n\n"
        f"  Sol Ayak:\n"
        f"    Shin length = {ls['shin_len']:.0f}\n"
        f"    max(nd)     = {ls['max_nd']:+.4f} → {classify(ls['max_nd'])}\n"
        f"    max(angle)  = {ls['max_angle']:+.1f}°\n\n"
        f"  Sağ Ayak:\n"
        f"    Shin length = {rs['shin_len']:.0f}\n"
        f"    max(nd)     = {rs['max_nd']:+.4f} → {classify(rs['max_nd'])}\n"
        f"    max(angle)  = {rs['max_angle']:+.1f}°\n\n"
        f"  Birincil:  mn_max  = {r['mn_max']:+.4f} → {r['type']}\n"
        f"  Doğrulama: mn_amax = {r['mn_amax']:+.1f}° → {r['type_angle']}\n"
        f"  Metrik uyumu: {agree}\n\n"
        f"  ► Sonuç: {r['type']} ({r['confidence']})\n\n"
        f"{'─'*46}\n"
        f"  Eşikler (birim-siz):\n"
        f"    > {TH_HEEL:+.3f} → Heel\n"
        f"    {TH_TOE:+.3f} ~ {TH_HEEL:+.3f} → Mid\n"
        f"    ≤ {TH_TOE:+.3f} → Toe\n"
        f"{'═'*46}"
    )
    ax.text(0.03, 0.97, summary, transform=ax.transAxes, fontsize=10,
            va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=.85))

    plt.tight_layout()
    plt.show()
    return r


# ── Toplu Analiz ──────────────────────────────────────────────────
def batch_analyze(folder="nlf_outputs"):
    files = sorted(f for f in os.listdir(folder) if f.endswith(".jsonl"))

    print(f"\n{'═'*95}")
    print("  TOPLU STRIKE ANALİZİ  (v4 – mn_max, çift metrik)")
    print(f"{'═'*95}\n")
    hdr = (f"  {'Dosya':<42} {'mn_max':>8} {'mn_amax':>8}"
           f"  {'Sonuç':<15} {'Doğr.':<15} {'Güven':<8}")
    print(hdr)
    print(f"  {'-'*91}")

    for f in files:
        r = analyze_strike(os.path.join(folder, f))
        agree = "✓" if r["type"] == r["type_angle"] else "✗"
        print(f"  {f:<42} {r['mn_max']:>+8.4f} {r['mn_amax']:>+7.1f}°"
              f"  {r['type']:<15} {r['type_angle']:<13}{agree} {r['confidence']:<8}")

    print(f"  {'-'*91}")
    print(f"\n  Eşikler:  mn_max > {TH_HEEL} → Heel  |"
          f"  {TH_TOE} < mn_max ≤ {TH_HEEL} → Mid  |"
          f"  ≤ {TH_TOE} → Toe")
    print(f"  Doğr.:    mn_amax > {TH_HEEL_A}° → Heel  |"
          f"  {TH_TOE_A}° < mn_amax ≤ {TH_HEEL_A}° → Mid  |"
          f"  ≤ {TH_TOE_A}° → Toe\n")


# ── CLI ───────────────────────────────────────────────────────────
def main():
    folder = "nlf_outputs"
    print("\n=== Heel / Toe / Mid Strike Analizi (v4) ===\n")
    print("  1: Tek dosya analizi (grafik)")
    print("  2: Toplu analiz (tüm dosyalar)")

    choice = input("\nSeçim (1/2): ").strip()

    if choice == "2":
        batch_analyze(folder)
    else:
        files = sorted(f for f in os.listdir(folder) if f.endswith(".jsonl"))
        for i, f in enumerate(files):
            print(f"  {i}: {f}")
        idx = int(input("\nDosya no: "))
        plot_strike(os.path.join(folder, files[idx]))


if __name__ == "__main__":
    main()
