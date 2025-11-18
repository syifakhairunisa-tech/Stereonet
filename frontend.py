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

# status
        self.status = tk.StringVar()
        self.status.set("Data: 0 pair (strike,dip)")
        tk.Label(root, textvariable=self.status).pack(side=tk.TOP, anchor="w", padx=6)

        # Area plot
        self.fig = None
        self.canvas = None
        self.plot_frame = tk.Frame(root)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

    def update_status(self):
        self.status.set(f"Data: {len(self.data)} pair (strike,dip)")

    def input_manual(self):
        # Dialog sederhana untuk input satu pair
        try:
            s = simpledialog.askfloat("Input Strike", "Strike (0-360):", minvalue=0.0, maxvalue=360.0)
            if s is None:
                return
            d = simpledialog.askfloat("Input Dip", "Dip (0-90):", minvalue=0.0, maxvalue=90.0)
            if d is None:
                return
            s2, d2 = backend.validate_strike_dip_pair(s, d)
            self.data.append((s2, d2))
            self.update_status()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files",".csv"),("All files",".*")])
        if not path:
            return
        try:
            data = backend.read_csv_file(path)
            # tambahkan ke data yang sudah ada
            self.data.extend(data)
            self.update_status()
            messagebox.showinfo("Success", f"Berhasil memuat {len(data)} pasangan dari {path}")
        except Exception as e:
            messagebox.showerror("Error membaca CSV", str(e))

    def clear_data(self):
        self.data = []
        self.update_status()
        self.clear_plot()

    def clear_plot(self):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
            self.fig = None

    def embed_figure(self, fig):
        self.clear_plot()
        self.fig = fig
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        widget = self.canvas.get_tk_widget()
        widget.pack(fill=tk.BOTH, expand=True)
# Handlers untuk plot
    def plot_stereonet(self):
        if not self.data:
            messagebox.showwarning("No data", "Belum ada data. Tambah lewat Input Manual atau Load CSV.")
            return
        fig = backend.plot_stereonet(self.data, title="Stereonet Plot (from GUI)")
        self.embed_figure(fig)

    def plot_rose(self):
        if not self.data:
            messagebox.showwarning("No data", "Belum ada data. Tambah lewat Input Manual atau Load CSV.")
            return
        fig = backend.plot_rose(self.data, bin_width=10, title="Rose Diagram (from GUI)")
        self.embed_figure(fig)

    def plot_density(self):
        if not self.data:
            messagebox.showwarning("No data", "Belum ada data. Tambah lewat Input Manual atau Load CSV.")
            return
        fig = backend.plot_polar_density(self.data, title="Polar Density Contour (from GUI)")
        self.embed_figure(fig)


if name == "main":
    root = tk.Tk()
    app = StereonetApp(root)
    root.geometry("800x700")
    root.mainloop()