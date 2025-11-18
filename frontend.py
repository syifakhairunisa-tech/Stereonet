"""
frontend.py
Contoh GUI sederhana (Tkinter) yang memanggil backend.py
Dependencies: tkinter, matplotlib, mplstereonet, backend.py, (pandas optional)
"""

import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog
from typing import List, Tuple
import backend  # pastikan backend.py ada di folder yang sama
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class StereonetApp:
    def _init_(self, root):
        self.root = root
        self.root.title("Stereonet GUI - Praktikum Metode Komputasi")
        self.data = []  # list of (strike, dip)

        # Frame kontrol
        ctl = tk.Frame(root)
        ctl.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)

        tk.Button(ctl, text="Input Manual", command=self.input_manual).pack(side=tk.LEFT, padx=4)
        tk.Button(ctl, text="Load CSV", command=self.load_csv).pack(side=tk.LEFT, padx=4)
        tk.Button(ctl, text="Clear Data", command=self.clear_data).pack(side=tk.LEFT, padx=4)
        tk.Label(ctl, text=" | ").pack(side=tk.LEFT)
        tk.Button(ctl, text="Plot Stereonet", command=self.plot_stereonet).pack(side=tk.LEFT, padx=4)
        tk.Button(ctl, text="Plot Rose", command=self.plot_rose).pack(side=tk.LEFT, padx=4)
        tk.Button(ctl, text="Plot Polar Density", command=self.plot_density).pack(side=tk.LEFT, padx=4)
