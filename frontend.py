"""
frontend.py
GUI Stereonet Project - Praktikum Metode Komputasi
Dependencies:
- tkinter
- matplotlib
- mplstereonet
- backend.py
"""

import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog
import backend  # pastikan backend.py berada di folder yang sama

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class StereonetApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stereonet GUI - Praktikum Metode Komputasi")
        self.data = []  # menampung daftar (strike, dip)

        # Frame kontrol (tombol)
        ctl = tk.Frame(root)
        ctl.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)

        tk.Button(ctl, text="Input Manual", command=self.input_manual).pack(side=tk.LEFT, padx=4)
        tk.Button(ctl, text="Load CSV", command=self.load_csv).pack(side=tk.LEFT, padx=4)
        tk.Button(ctl, text="Clear Data", command=self.clear_data).pack(side=tk.LEFT, padx=4)
        tk.Label(ctl, text=" | ").pack(side=tk.LEFT)
        tk.Button(ctl, text="Plot Stereonet", command=self.plot_stereonet).pack(side=tk.LEFT, padx=4)
        tk.Button(ctl, text="Plot Rose", command=self.plot_rose).pack(side=tk.LEFT, padx=4)
        tk.Button(ctl, text="Plot Polar Density", command=self.plot_density).pack(side=tk.LEFT, padx=4)

        # Status jumlah data
        self.status = tk.StringVar()
        self.status.set("Data: 0 pair (strike, dip)")
        tk.Label(root, textvariable=self.status).pack(side=tk.TOP, anchor="w", padx=6)

        # Area untuk plot gambar
        self.fig = None
        self.canvas = None
        self.plot_frame = tk.Frame(root)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

    # Update status jumlah data
    def update_status(self):
        self.status.set(f"Data: {len(self.data)} pair (strike, dip)")

    # Input manual dalam 1 dialog
    def input_manual(self):
        win = tk.Toplevel(self.root)
        win.title("Input Manual Strike & Dip")
        win.geometry("250x160")
        win.resizable(False, False)

        tk.Label(win, text="Strike (0–360):").pack(pady=4)
        strike_entry = tk.Entry(win)
        strike_entry.pack()

        tk.Label(win, text="Dip (0–90):").pack(pady=4)
        dip_entry = tk.Entry(win)
        dip_entry.pack()

        def submit():
            try:
                s = float(strike_entry.get())
                d = float(dip_entry.get())
                s2, d2 = backend.validate_strike_dip_pair(s, d)
                self.data.append((s2, d2))
                self.update_status()
                win.destroy()
            except Exception as e:
                messagebox.showerror("Input Error", str(e))

        tk.Button(win, text="OK", command=submit).pack(pady=10)

    # Load file CSV
    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"),("All files", "*.*")])
        if not path:
            return

        try:
            data = backend.read_csv_file(path)
            self.data.extend(data)
            self.update_status()
            messagebox.showinfo("Success", f"Berhasil memuat {len(data)} data dari file!")
        except Exception as e:
            messagebox.showerror("Error membaca CSV", str(e))

    # Clear data & tampilan plot
    def clear_data(self):
        self.data = []
        self.update_status()
        self.clear_plot()

    def clear_plot(self):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
            self.fig = None

    # Embed figure matplotlib ke GUI Tkinter
    def embed_figure(self, fig):
        self.clear_plot()
        self.fig = fig
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Handler tombol plot
    def plot_stereonet(self):
        if not self.data:
            messagebox.showwarning("No Data", "Tambahkan data terlebih dahulu.")
            return
        fig = backend.plot_stereonet(self.data, title="Stereonet Plot")
        self.embed_figure(fig)

    def plot_rose(self):
        if not self.data:
            messagebox.showwarning("No Data", "Tambahkan data terlebih dahulu.")
            return
        fig = backend.plot_rose(self.data, bin_width=10, title="Rose Diagram")
        self.embed_figure(fig)

    def plot_density(self):
        if not self.data:
            messagebox.showwarning("No Data", "Tambahkan data terlebih dahulu.")
            return
        fig = backend.plot_polar_density(self.data, title="Polar Density Contour")
        self.embed_figure(fig)


# Main loop
if __name__ == "__main__":
    root = tk.Tk()
    app = StereonetApp(root)
    root.geometry("850x720")
    root.mainloop()