"""
backend.py
Modul backend untuk project Stereonet:
- Mengandung rumus-rumus geologi (true/apparent dip, dip direction)
- Fungsi utilitas (normalisasi sudut, validasi data, baca CSV)
- Wrapper plotting (stereonet, rose diagram, polar density contour)
Dependencies:
    numpy, matplotlib, mplstereonet, pandas (opsional untuk baca csv)
"""

from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import mplstereonet
import math
import csv
import io

try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False

# Type alias: list of (strike, dip) in degrees
SData = List[Tuple[float, float]]


# BAGIAN A: UTILITAS ANGLE / VALIDASI
def normalize_angle(angle: float) -> float:
    """Normalisasi sudut ke rentang [0, 360)."""
    return angle % 360


def validate_strike_dip_pair(strike: float, dip: float) -> Tuple[float, float]:
    """
    Validasi satu pasangan strike/dip dan normalisasi:
    - strike: 0 <= strike < 360
    - dip: 0 <= dip <= 90 (biasanya 0..90)
    Mengembalikan tuple (strike_norm, dip_norm) atau raise ValueError.
    """
    if not (0 <= dip <= 90):
        raise ValueError(f"Dip harus di antara 0 dan 90 derajat, diberikan: {dip}")
    strike_n = normalize_angle(float(strike))
    return strike_n, float(dip)


def validate_dataset(data: SData) -> SData:
    """Validasi list pasangan (strike, dip)."""
    validated = []
    for i, (s, d) in enumerate(data):
        try:
            s2, d2 = validate_strike_dip_pair(s, d)
            validated.append((s2, d2))
        except Exception as e:
            raise ValueError(f"Data index {i} invalid: {e}")
    return validated


# BAGIAN B: RUMUS GEOLOGI (TRUE / APPARENT DIP, DIP DIRECTION)
def true_dip(apparent_dip: float, angle_between_strikes: float) -> float:
    """
    Hitung True Dip (δ_t) dari Apparent Dip (δ_a) dan sudut α :
        tan(δ_t) = tan(δ_a) / cos(α)

    Input/output dalam derajat.
    angle_between_strikes (α) = sudut antara arah apparent dip dan arah true dip (derajat).
    """
    # Konversi ke radian
    da = math.radians(apparent_dip)
    a = math.radians(angle_between_strikes)
    # Hindari pembagian dengan nol (cos(a) = 0)
    cos_a = math.cos(a)
    if abs(cos_a) < 1e-12:
        raise ValueError("cos(angle_between_strikes) terlalu kecil -> hasil akan tak terhingga")
    dt_rad = math.atan(math.tan(da) / cos_a)
    return math.degrees(dt_rad)


def apparent_dip(true_dip_val: float, angle_between_strikes: float) -> float:
    """
    Hitung Apparent Dip (δ_a) dari True Dip (δ_t) dan sudut α:
        tan(δ_a) = tan(δ_t) * cos(α)
    Input/output dalam derajat.
    """
    dt = math.radians(true_dip_val)
    a = math.radians(angle_between_strikes)
    da_rad = math.atan(math.tan(dt) * math.cos(a))
    return math.degrees(da_rad)


def dip_direction_from_strike(strike: float) -> float:
    """
    Hitung arah dip (dip direction) dari strike menggunakan konvensi right-hand rule:
        dip_direction = (strike + 90) % 360
    Return dalam derajat [0,360).
    """
    return normalize_angle(strike + 90)


# BAGIAN C: I/O DATA (CSV)
def read_csv_string(csv_text: str, strike_col: str = 'strike', dip_col: str = 'dip') -> SData:
    """
    Baca dataset dari string CSV. Baris pertama dapat berisi header.
    Kolom default: 'strike', 'dip'. Bisa disesuaikan.
    Mengembalikan list (strike, dip) setelah validasi.
    """
    # coba pakai pandas jika ada
    if _HAS_PANDAS:
        df = pd.read_csv(io.StringIO(csv_text))
        if strike_col not in df.columns or dip_col not in df.columns:
            raise ValueError(f"CSV harus memiliki kolom '{strike_col}' dan '{dip_col}'")
        data = list(zip(df[strike_col].astype(float).tolist(), df[dip_col].astype(float).tolist()))
        return validate_dataset(data)

    # fallback ke csv module
    reader = csv.DictReader(io.StringIO(csv_text))
    data = []
    for row in reader:
        if strike_col not in row or dip_col not in row:
            raise ValueError(f"CSV harus memiliki header kolom '{strike_col}' dan '{dip_col}'")
        s = float(row[strike_col])
        d = float(row[dip_col])
        data.append((s, d))
    return validate_dataset(data)


def read_csv_file(path: str, strike_col: str = 'strike', dip_col: str = 'dip') -> SData:
    """Baca file CSV dari path (mengembalikan list (strike,dip))."""
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return read_csv_string(text, strike_col=strike_col, dip_col=dip_col)


# BAGIAN D: PLOTTING WRAPPERS
# Semua fungsi plotting mengembalikan objek Figure.

def plot_stereonet(data: SData, title: str = "Stereonet Plot") -> plt.Figure:
    """
    Plot bidang (plane) dan kutub (pole) pada stereonet.
    data: list of (strike, dip) in degrees.
    Mengembalikan matplotlib.figure.Figure.
    """
    data = validate_dataset(data)
    fig = plt.figure(figsize=(6.5, 6.5))
    ax = fig.add_subplot(111, projection='stereonet')

    for strike, dip in data:
        # plot bidang
        ax.plane(strike, dip, linewidth=1, color='C0', alpha=0.8)
        # plot kutub bidang (mplstereonet menghitungnya sesuai konvensi)
        ax.pole(strike, dip, marker='o', markersize=6, color='C1')

    ax.grid(True)
    ax.set_title(title, pad=20)
    return fig


def plot_rose(data: SData, bin_width: int = 10, title: str = "Rose Diagram") -> plt.Figure:
    """
    Plot rose diagram (histogram polar) berdasarkan nilai strike.
    bin_width dalam derajat (mis. 10).
    """
    data = validate_dataset(data)
    strikes = np.array([s for s, d in data])
    # Convert to radians
    strikes_rad = np.radians(strikes)

    bins_deg = np.arange(0, 360 + bin_width, bin_width)
    bins = np.radians(bins_deg)
    counts, _ = np.histogram(strikes_rad, bins=bins)

    fig = plt.figure(figsize=(6.5, 6.5))
    ax = fig.add_subplot(111, polar=True)
    # align bars on center of each bin; bars use bins[:-1] as left edge -> shift by bin_width/2
    left = bins[:-1]
    width = np.radians(bin_width)

    ax.bar(left + width/2, counts, width=width, bottom=0.0, align='center', edgecolor='k', alpha=0.7)
    ax.set_theta_zero_location("N")  # 0° at top
    ax.set_theta_direction(-1)       # clockwise
    ax.set_title(title, pad=20)
    return fig


def plot_polar_density(data: SData, title: str = "Polar Density Contour (Poles)") -> plt.Figure:
    """
    Plot polar density contour (KDE) dari kutub bidang.
    Menggunakan fungsi bawaan dari mplstereonet.
    """
    data = validate_dataset(data)
    strikes = [s for s, d in data]
    dips = [d for s, d in data]
    fig = plt.figure(figsize=(6.5, 6.5))
    ax = fig.add_subplot(111, projection='stereonet')

    # density_contourf akan melakukan smoothing (KDE) pada kutub bidang
    ax.density_contourf(strikes, dips, measurement='poles', cmap='viridis', sigma=10)
    ax.density_contour(strikes, dips, measurement='poles', colors='k', linewidths=0.5)
    ax.set_title(title, pad=20)
    return fig


# BAGIAN E: HELPERS / EXAMPLES
def example_dataset() -> SData:
    """Menghasilkan dataset contoh kecil untuk testing."""
    return [
        (315, 30),
        (120, 25),
        (220, 40),
        (60, 15),
        (130, 50),
        (310, 35),
    ]


if _name_ == "_main_":
    # Demo singkat bila dijalankan langsung
    data = example_dataset()
    print("Contoh dataset:", data)
    print("Contoh true_dip(25 pada angle 30):", true_dip(25, 30))

    fig = plot_stereonet(data)
    fig.show()
    fig2 = plot_rose(data)
    fig2.show()
    fig3 = plot_polar_density(data)
    fig3.show()