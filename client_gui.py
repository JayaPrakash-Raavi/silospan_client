import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import client

class ThreadSafeConsole:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, text):
        self.text_widget.after(0, self._write_to_widget, text)

    def _write_to_widget(self, text):
        self.text_widget.insert("end", text)
        self.text_widget.see("end")

    def flush(self):
        pass

class SiloSpanClientGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SiloSpan Federated Client")
        self.root.geometry("900x550")
        self.root.configure(bg="#121212")
        
        # Configure modern look
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure(".", background="#121212", foreground="#FFFFFF", fieldbackground="#1e1e1e")
        self.style.configure("TLabel", background="#121212", foreground="#E0E0E0", font=("Segoe UI", 10))
        self.style.configure("TButton", background="#3f51b5", foreground="#FFFFFF", font=("Segoe UI", 10, "bold"), borderwidth=0)
        self.style.map("TButton", background=[("active", "#303f9f")])
        self.style.configure("TEntry", fieldbackground="#1e1e1e", foreground="#FFFFFF", insertcolor="#FFFFFF")
        
        # Main layout: Left sidebar for config, Right for logs
        self.left_frame = tk.Frame(self.root, bg="#121212", width=350, padx=20, pady=20)
        self.left_frame.pack(side="left", fill="y")
        self.left_frame.pack_propagate(False)
        
        self.right_frame = tk.Frame(self.root, bg="#1a1a1a", padx=10, pady=10)
        self.right_frame.pack(side="right", fill="both", expand=True)
        
        # --- Config Fields (Left Side) ---
        tk.Label(self.left_frame, text="SILOSPAN CLIENT CONFIG", bg="#121212", fg="#3f51b5", font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(0, 20))
        
        # Hub Address
        ttk.Label(self.left_frame, text="Hub Domain Address:").pack(anchor="w", pady=(5, 2))
        self.addr_entry = ttk.Entry(self.left_frame, width=40)
        self.addr_entry.insert(0, "silospan.sabyasacheemishra.com:8080")
        self.addr_entry.pack(anchor="w", fill="x", pady=2)
        
        # API Key
        ttk.Label(self.left_frame, text="Client API Key:").pack(anchor="w", pady=(10, 2))
        self.key_entry = ttk.Entry(self.left_frame, show="*", width=40)
        self.key_entry.insert(0, "silospan_client_secret_key_2026")
        self.key_entry.pack(anchor="w", fill="x", pady=2)
        
        # Local Partition ID
        ttk.Label(self.left_frame, text="Local Partition ID:").pack(anchor="w", pady=(10, 2))
        self.part_entry = ttk.Entry(self.left_frame, width=40)
        self.part_entry.insert(0, "0")
        self.part_entry.pack(anchor="w", fill="x", pady=2)
        
        # Total Partitions
        ttk.Label(self.left_frame, text="Total Partitions:").pack(anchor="w", pady=(10, 2))
        self.tot_entry = ttk.Entry(self.left_frame, width=40)
        self.tot_entry.insert(0, "2")
        self.tot_entry.pack(anchor="w", fill="x", pady=2)
        
        # Differential Privacy (DP) Settings
        tk.Label(self.left_frame, text="Differential Privacy", bg="#121212", fg="#888888", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(15, 5))
        
        dp_frame = tk.Frame(self.left_frame, bg="#121212")
        dp_frame.pack(fill="x")
        
        ttk.Label(dp_frame, text="Sigma (Noise):").grid(row=0, column=0, sticky="w", pady=2)
        self.sigma_entry = ttk.Entry(dp_frame, width=10)
        self.sigma_entry.insert(0, "0.01")
        self.sigma_entry.grid(row=0, column=1, sticky="w", padx=(10, 0), pady=2)
        
        ttk.Label(dp_frame, text="Clipping Norm:").grid(row=1, column=0, sticky="w", pady=2)
        self.clip_entry = ttk.Entry(dp_frame, width=10)
        self.clip_entry.insert(0, "1.0")
        self.clip_entry.grid(row=1, column=1, sticky="w", padx=(10, 0), pady=2)
        
        # Start button
        self.start_btn = tk.Button(
            self.left_frame, text="CONNECT & START TRAINING", 
            command=self.start_training_thread,
            bg="#3f51b5", fg="#FFFFFF", font=("Segoe UI", 11, "bold"),
            activebackground="#303f9f", activeforeground="#FFFFFF",
            relief="flat", bd=0, cursor="hand2", height=2
        )
        self.start_btn.pack(side="bottom", fill="x", pady=(20, 0))
        
        # --- Terminal/Log Area (Right Side) ---
        tk.Label(self.right_frame, text="REAL-TIME TRAINING LOGS", bg="#1a1a1a", fg="#888888", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 5))
        
        self.log_text = tk.Text(self.right_frame, bg="#0d0d0d", fg="#a6e22e", font=("Consolas", 9), wrap="word", bd=0)
        self.log_text.pack(side="left", fill="both", expand=True)
        
        self.scrollbar = ttk.Scrollbar(self.right_frame, orient="vertical", command=self.log_text.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=self.scrollbar.set)
        
        # Redirect stdout and stderr to the log terminal
        sys.stdout = ThreadSafeConsole(self.log_text)
        sys.stderr = ThreadSafeConsole(self.log_text)
        
        print("[SYSTEM] SiloSpan Client GUI Initialized.")
        print("[SYSTEM] Configure settings and click 'CONNECT & START TRAINING' to participate.")

    def start_training_thread(self):
        # Validate inputs
        try:
            part = int(self.part_entry.get().strip())
            tot = int(self.tot_entry.get().strip())
            sigma = float(self.sigma_entry.get().strip())
            clip = float(self.clip_entry.get().strip())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers for Partition ID, Total Partitions, and DP variables.")
            return
            
        addr = self.addr_entry.get().strip()
        key = self.key_entry.get().strip()
        
        if not addr:
            messagebox.showerror("Invalid Address", "Hub Domain Address cannot be blank.")
            return

        self.start_btn.configure(state="disabled", text="RUNNING FL SESSION...", bg="#555555")
        
        # Run FL client in a background thread to prevent UI freezing
        t = threading.Thread(
            target=self.run_client_worker,
            args=(addr, part, tot, sigma, clip, key),
            daemon=True
        )
        t.start()

    def run_client_worker(self, addr, part, tot, sigma, clip, key):
        try:
            client.start_client(
                server_address=addr,
                partition=part,
                total_partitions=tot,
                epochs=1,
                lr=0.01,
                dp_sigma=sigma,
                dp_clipping=clip,
                device="cpu",
                ssl_ca="certs/ca.crt",
                api_key=key
            )
        except Exception as e:
            print(f"\n[GUI ERROR] Failed: {e}")
        finally:
            self.root.after(0, self.reset_start_button)

    def reset_start_button(self):
        self.start_btn.configure(state="normal", text="CONNECT & START TRAINING", bg="#3f51b5")

if __name__ == "__main__":
    root = tk.Tk()
    app = SiloSpanClientGUI(root)
    root.mainloop()
