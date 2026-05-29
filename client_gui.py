import os
import sys
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import client

class ThreadSafeConsole:
    def __init__(self, text_widget, queue_obj):
        self.text_widget = text_widget
        self.queue = queue_obj

    def write(self, text):
        # Place log event on the queue to be processed safely by the main thread
        self.queue.put(("log", text))

    def flush(self):
        pass

class MetricCanvasChart(tk.Canvas):
    def __init__(self, parent, width=440, height=240, bg="#1E293B", highlightthickness=0):
        super().__init__(parent, width=width, height=height, bg=bg, highlightthickness=highlightthickness)
        self.width = width
        self.height = height
        self.loss_history = []
        self.accuracy_history = []
        self.draw_empty_chart()

    def draw_empty_chart(self):
        self.delete("all")
        grid_color = "#334155"
        self.padx = 45
        self.pady = 25
        
        # Draw axes
        self.create_line(self.padx, self.pady, self.padx, self.height - self.pady, fill=grid_color, width=1)
        self.create_line(self.padx, self.height - self.pady, self.width - self.padx, self.height - self.pady, fill=grid_color, width=1)
        self.create_line(self.width - self.padx, self.pady, self.width - self.padx, self.height - self.pady, fill=grid_color, width=1)
        
        # Grid lines
        plot_h = self.height - 2 * self.pady
        for i in range(1, 4):
            y = self.pady + plot_h * (i / 4.0)
            self.create_line(self.padx, y, self.width - self.padx, y, fill=grid_color, dash=(2, 2))
            
        self.create_text(self.width // 2, self.height // 2, text="Live training chart will plot here", fill="#64748B", font=("Segoe UI", 9, "italic"))
        self.create_text(self.padx + 30, self.pady - 10, text="Loss", fill="#F87171", font=("Segoe UI", 8, "bold"))
        self.create_text(self.width - self.padx - 30, self.pady - 10, text="Accuracy", fill="#34D399", font=("Segoe UI", 8, "bold"))

    def update_metrics(self, loss_history, accuracy_history):
        self.loss_history = loss_history
        self.accuracy_history = accuracy_history
        self.redraw()

    def redraw(self):
        self.delete("all")
        grid_color = "#334155"
        self.padx = 45
        self.pady = 25
        
        plot_w = self.width - 2 * self.padx
        plot_h = self.height - 2 * self.pady
        
        # Draw axes
        self.create_line(self.padx, self.pady, self.padx, self.height - self.pady, fill=grid_color, width=1)
        self.create_line(self.padx, self.height - self.pady, self.width - self.padx, self.height - self.pady, fill=grid_color, width=1)
        self.create_line(self.width - self.padx, self.pady, self.width - self.padx, self.height - self.pady, fill=grid_color, width=1)
        
        n_points = len(self.loss_history)
        if n_points == 0:
            self.draw_empty_chart()
            return
            
        # Draw horizontal grid lines
        for i in range(5):
            val_percent = i / 4.0
            y = self.height - self.pady - plot_h * val_percent
            self.create_line(self.padx, y, self.width - self.padx, y, fill="#1E293B" if i == 0 else "#2D3748", dash=(1, 2))
            
        # Scaling factors for Loss (Left Axis)
        max_loss = max(self.loss_history) if self.loss_history else 1.0
        min_loss = min(self.loss_history) if self.loss_history else 0.0
        loss_range = max_loss - min_loss if max_loss != min_loss else 1.0
        
        # Loss labels (left side, Coral Red)
        for i in range(5):
            val_percent = i / 4.0
            y = self.height - self.pady - plot_h * val_percent
            loss_val = min_loss + loss_range * val_percent
            self.create_text(self.padx - 8, y, text=f"{loss_val:.2f}", fill="#F87171", anchor="e", font=("Consolas", 8))
            
        # Accuracy labels (right side, Emerald Green)
        for i in range(5):
            val_percent = i / 4.0
            y = self.height - self.pady - plot_h * val_percent
            acc_val = val_percent * 100
            self.create_text(self.width - self.padx + 8, y, text=f"{acc_val:.0f}%", fill="#34D399", anchor="w", font=("Consolas", 8))

        # Legends
        self.create_text(self.padx + 30, self.pady - 10, text="Loss (Red)", fill="#F87171", font=("Segoe UI", 9, "bold"))
        self.create_text(self.width - self.padx - 40, self.pady - 10, text="Accuracy (Green)", fill="#34D399", font=("Segoe UI", 9, "bold"))
        
        # Plot Loss
        loss_pts = []
        for idx, loss in enumerate(self.loss_history):
            x = self.padx if n_points == 1 else self.padx + plot_w * (idx / (n_points - 1))
            norm_loss = (loss - min_loss) / loss_range
            y = self.height - self.pady - plot_h * norm_loss
            loss_pts.append((x, y))
            self.create_oval(x-3, y-3, x+3, y+3, fill="#EF4444", outline="#F87171")
            
        if len(loss_pts) > 1:
            for k in range(len(loss_pts)-1):
                self.create_line(loss_pts[k][0], loss_pts[k][1], loss_pts[k+1][0], loss_pts[k+1][1], fill="#EF4444", width=2)

        # Plot Accuracy (Accuracy is passed as float 0.0 to 1.0)
        acc_pts = []
        for idx, acc in enumerate(self.accuracy_history):
            x = self.padx if n_points == 1 else self.padx + plot_w * (idx / (n_points - 1))
            y = self.height - self.pady - plot_h * acc
            acc_pts.append((x, y))
            self.create_oval(x-3, y-3, x+3, y+3, fill="#10B981", outline="#34D399")
            
        if len(acc_pts) > 1:
            for k in range(len(acc_pts)-1):
                self.create_line(acc_pts[k][0], acc_pts[k][1], acc_pts[k+1][0], acc_pts[k+1][1], fill="#10B981", width=2)

class PulsingLED(tk.Canvas):
    def __init__(self, parent, size=16, bg="#0F172A", highlightthickness=0):
        super().__init__(parent, width=size, height=size, bg=bg, highlightthickness=highlightthickness)
        self.size = size
        self.color = "#64748B"  # Default Gray
        self.pulse_state = True
        self.animate()
        
    def set_color(self, color):
        self.color = color
        self.redraw()
        
    def redraw(self):
        self.delete("all")
        margin = 2
        color = self.color
        
        # Simulating pulsing for active states
        if not self.pulse_state:
            if self.color == "#F59E0B":    # Training (amber)
                color = "#78350F"
            elif self.color == "#3B82F6":  # Connecting (blue)
                color = "#1E3A8A"
            elif self.color == "#10B981":  # Evaluating (emerald green)
                color = "#064E3B"
                
        self.create_oval(margin, margin, self.size - margin, self.size - margin, fill=color, outline="#1E293B", width=1)
        # Reflection highlight
        self.create_oval(margin + 2, margin + 2, margin + 5, margin + 5, fill="#FFFFFF", outline="")
        
    def animate(self):
        self.pulse_state = not self.pulse_state
        self.redraw()
        self.after(500, self.animate)

class ModernProgressBar(tk.Canvas):
    def __init__(self, parent, height=10, bg="#1E293B", bar_color="#3B82F6", highlightthickness=0):
        super().__init__(parent, height=height, bg=bg, highlightthickness=highlightthickness)
        self.height = height
        self.bar_color = bar_color
        self.progress = 0.0
        self.bind("<Configure>", lambda e: self.redraw())
        
    def set_progress(self, progress):
        self.progress = max(0.0, min(1.0, progress))
        self.redraw()
        
    def redraw(self):
        self.delete("all")
        w = self.winfo_width()
        if w < 1:
            return
        self.create_rectangle(0, 0, w * self.progress, self.height, fill=self.bar_color, outline="")

class CustomEntry(tk.Frame):
    def __init__(self, parent, label_text, default_value="", show=None, width=30):
        super().__init__(parent, bg="#0F172A", pady=5)
        
        lbl = tk.Label(self, text=label_text, fg="#94A3B8", bg="#0F172A", font=("Segoe UI", 9, "bold"), anchor="w")
        lbl.pack(fill="x", anchor="w")
        
        border_frame = tk.Frame(self, bg="#334155", bd=1)
        border_frame.pack(fill="x", anchor="w", pady=(2, 0))
        
        self.entry = tk.Entry(
            border_frame, 
            bg="#1E293B", 
            fg="#F8FAFC", 
            insertbackground="#F8FAFC",
            relief="flat", 
            bd=4, 
            width=width,
            font=("Segoe UI", 10),
            show=show
        )
        self.entry.insert(0, default_value)
        self.entry.pack(fill="x")
        
    def get(self):
        return self.entry.get()
        
    def set(self, val):
        self.entry.delete(0, tk.END)
        self.entry.insert(0, val)

    def set_state(self, state):
        self.entry.configure(state=state)

class PremiumButton(tk.Button):
    def __init__(self, parent, text, command=None, bg="#3B82F6", fg="#FFFFFF", activebg="#2563EB", activefg="#FFFFFF", **kwargs):
        super().__init__(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=activebg,
            activeforeground=activefg,
            relief="flat",
            bd=0,
            cursor="hand2",
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=6,
            **kwargs
        )
        self.bg = bg
        self.activebg = activebg
        self.bind("<Enter>", lambda e: self.configure(bg=self.activebg) if self["state"] != "disabled" else None)
        self.bind("<Leave>", lambda e: self.configure(bg=self.bg) if self["state"] != "disabled" else None)

class MetricCard(tk.Frame):
    def __init__(self, parent, title, value, color="#3B82F6"):
        super().__init__(parent, bg="#1E293B", padx=15, pady=10, highlightbackground="#334155", highlightthickness=1)
        self.lbl_title = tk.Label(self, text=title.upper(), fg="#94A3B8", bg="#1E293B", font=("Segoe UI", 8, "bold"))
        self.lbl_title.pack(anchor="w")
        self.lbl_val = tk.Label(self, text=value, fg=color, bg="#1E293B", font=("Segoe UI", 14, "bold"))
        self.lbl_val.pack(anchor="w", pady=(4, 0))
        
    def set_value(self, value):
        self.lbl_val.configure(text=value)

class TabNavigation(tk.Frame):
    def __init__(self, parent, initial_tab=0, **kwargs):
        super().__init__(parent, bg="#0F172A", **kwargs)
        self.tabs_config = []
        self.buttons = []
        self.active_tab = initial_tab
        
        self.btn_bar = tk.Frame(self, bg="#0F172A")
        self.btn_bar.pack(fill="x", anchor="w")
        
        line = tk.Frame(self, bg="#1E293B", height=2)
        line.pack(fill="x", anchor="w", pady=(0, 10))
        
        self.content_container = tk.Frame(self, bg="#0F172A")
        self.content_container.pack(fill="both", expand=True)
        
    def add_tab(self, label, frame):
        idx = len(self.tabs_config)
        btn = tk.Button(
            self.btn_bar,
            text=label,
            command=lambda: self.select_tab(idx),
            bg="#0F172A",
            fg="#94A3B8",
            activebackground="#0F172A",
            activeforeground="#F8FAFC",
            relief="flat",
            bd=0,
            cursor="hand2",
            font=("Segoe UI", 10, "bold"),
            padx=20,
            pady=8
        )
        btn.pack(side="left")
        self.buttons.append(btn)
        self.tabs_config.append((label, frame))
        frame.pack_forget()
        
    def select_tab(self, tab_idx):
        if tab_idx < 0 or tab_idx >= len(self.tabs_config):
            return
            
        if len(self.tabs_config) > self.active_tab:
            self.buttons[self.active_tab].configure(fg="#94A3B8")
            self.tabs_config[self.active_tab][1].pack_forget()
            
        self.active_tab = tab_idx
        self.buttons[tab_idx].configure(fg="#3B82F6")
        self.tabs_config[tab_idx][1].pack(fill="both", expand=True)

class SiloSpanClientGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SiloSpan Edge Client Control Panel")
        self.root.geometry("1024x600")
        self.root.configure(bg="#0F172A")
        
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.client_thread = None
        
        # Telemetry metrics history
        self.loss_history = []
        self.accuracy_history = []
        
        # Top Header Bar
        header = tk.Frame(self.root, bg="#0F172A", pady=10, padx=20)
        header.pack(fill="x")
        
        title_frame = tk.Frame(header, bg="#0F172A")
        title_frame.pack(side="left")
        
        tk.Label(title_frame, text="SILOSPAN", fg="#F8FAFC", bg="#0F172A", font=("Segoe UI", 14, "bold")).pack(side="left")
        badge = tk.Frame(title_frame, bg="#1E3A8A", padx=6, pady=2)
        badge.pack(side="left", padx=(10, 0))
        tk.Label(badge, text="EDGE CLIENT", fg="#60A5FA", bg="#1E3A8A", font=("Segoe UI", 8, "bold")).pack()
        
        # Right Header telemetry status indicator
        status_panel = tk.Frame(header, bg="#0F172A")
        status_panel.pack(side="right")
        
        self.status_led = PulsingLED(status_panel, size=14, bg="#0F172A")
        self.status_led.pack(side="left", padx=(0, 6), pady=4)
        
        self.status_lbl = tk.Label(status_panel, text="STATUS: IDLE", fg="#94A3B8", bg="#0F172A", font=("Segoe UI", 9, "bold"))
        self.status_lbl.pack(side="right")

        # Add Tabs wrapper
        self.navigation = TabNavigation(self.root, initial_tab=0)
        self.navigation.pack(fill="both", expand=True, padx=20)

        # Create the sub-frames for Tabs
        self.dashboard_frame = tk.Frame(self.navigation.content_container, bg="#0F172A", padx=20, pady=10)
        self.config_frame = tk.Frame(self.navigation.content_container, bg="#0F172A", padx=20, pady=10)
        self.logs_frame = tk.Frame(self.navigation.content_container, bg="#0F172A", padx=20, pady=10)
        
        # Init components
        self.setup_dashboard_tab()
        self.setup_config_tab()
        self.setup_logs_tab()
        
        # Add Tabs to navigation
        self.navigation.add_tab("DASHBOARD", self.dashboard_frame)
        self.navigation.add_tab("CONFIGURATION", self.config_frame)
        self.navigation.add_tab("SYSTEM CONSOLE", self.logs_frame)
        
        self.navigation.select_tab(0)
        
        # Redirect stdout and stderr using safe logs queue
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = ThreadSafeConsole(self.log_text, self.queue)
        sys.stderr = ThreadSafeConsole(self.log_text, self.queue)
        
        print("[SYSTEM] SiloSpan Edge Node Dashboard Initialized.")
        print("[SYSTEM] Default configuration generated. Ready to connect.")
        
        # Start queue poller
        self.poll_queue()

    def setup_dashboard_tab(self):
        # Two columns layout: Left side for node details & action, Right side for charts
        left_col = tk.Frame(self.dashboard_frame, bg="#0F172A")
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        right_col = tk.Frame(self.dashboard_frame, bg="#0F172A")
        right_col.pack(side="right", fill="both", expand=True, padx=(10, 0))
        
        # Left Panel content: Telemetry metrics lists
        section_lbl = tk.Label(left_col, text="NODE RUNTIME TELEMETRY", fg="#3B82F6", bg="#0F172A", font=("Segoe UI", 10, "bold"))
        section_lbl.pack(anchor="w", pady=(0, 10))
        
        telemetry_container = tk.Frame(left_col, bg="#1E293B", padx=15, pady=15, highlightbackground="#334155", highlightthickness=1)
        telemetry_container.pack(fill="x", anchor="w")
        
        # Telemetry detail rows helper
        self.telemetry_rows = {}
        fields = [
            ("server", "Hub Address:", "silospan.sabyasacheemishra.com:8080"),
            ("partition", "Dataset Partition:", "Partition 0 of 2"),
            ("device", "Computation Device:", "CPU"),
            ("dp", "Differential Privacy:", "Disabled"),
            ("round_info", "Current Round:", "None"),
        ]
        
        for key, label, default in fields:
            row = tk.Frame(telemetry_container, bg="#1E293B", pady=5)
            row.pack(fill="x")
            tk.Label(row, text=label, fg="#94A3B8", bg="#1E293B", font=("Segoe UI", 9, "bold")).pack(side="left")
            val_lbl = tk.Label(row, text=default, fg="#F8FAFC", bg="#1E293B", font=("Segoe UI", 9))
            val_lbl.pack(side="right")
            self.telemetry_rows[key] = val_lbl
            
        # Progress section
        progress_lbl = tk.Label(left_col, text="TRAINING PROGRESS", fg="#3B82F6", bg="#0F172A", font=("Segoe UI", 10, "bold"))
        progress_lbl.pack(anchor="w", pady=(15, 5))
        
        progress_container = tk.Frame(left_col, bg="#1E293B", padx=15, pady=15, highlightbackground="#334155", highlightthickness=1)
        progress_container.pack(fill="x", anchor="w")
        
        # Epoch Progress
        self.epoch_prog_lbl = tk.Label(progress_container, text="Epoch Progress: 0%", fg="#94A3B8", bg="#1E293B", font=("Segoe UI", 9, "bold"))
        self.epoch_prog_lbl.pack(anchor="w")
        self.epoch_progress_bar = ModernProgressBar(progress_container, height=8, bg="#0F172A", bar_color="#3B82F6")
        self.epoch_progress_bar.pack(fill="x", pady=(5, 10))
        
        # Overall Rounds Progress
        self.round_prog_lbl = tk.Label(progress_container, text="Training Status: Waiting to start", fg="#94A3B8", bg="#1E293B", font=("Segoe UI", 9, "bold"))
        self.round_prog_lbl.pack(anchor="w")
        
        # Buttons panel at the bottom
        btn_panel = tk.Frame(left_col, bg="#0F172A")
        btn_panel.pack(fill="x", side="bottom", pady=(20, 0))
        
        self.start_btn = PremiumButton(
            btn_panel, text="START PARTICIPATING", command=self.start_training,
            bg="#10B981", fg="#FFFFFF", activebg="#059669"
        )
        self.start_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        self.stop_btn = PremiumButton(
            btn_panel, text="STOP CLIENT", command=self.stop_training,
            bg="#EF4444", fg="#FFFFFF", activebg="#DC2626"
        )
        self.stop_btn.pack(side="right", fill="x", expand=True, padx=(5, 0))
        self.stop_btn.configure(state="disabled", bg="#334155")

        # Right Panel content: Visual metrics charts & summary cards
        chart_section_lbl = tk.Label(right_col, text="REAL-TIME TRAINING LOSS & ACCURACY", fg="#3B82F6", bg="#0F172A", font=("Segoe UI", 10, "bold"))
        chart_section_lbl.pack(anchor="w", pady=(0, 10))
        
        # Live Chart
        self.metrics_chart = MetricCanvasChart(right_col, bg="#1E293B", width=480, height=240)
        self.metrics_chart.pack(fill="x")
        
        # Metric Summary Cards side-by-side
        cards_frame = tk.Frame(right_col, bg="#0F172A", pady=15)
        cards_frame.pack(fill="x")
        
        self.acc_card = MetricCard(cards_frame, "Round Accuracy", "0.00%", "#10B981")
        self.acc_card.pack(side="left", fill="both", expand=True, padx=(0, 8))
        
        self.loss_card = MetricCard(cards_frame, "Round Loss", "0.0000", "#EF4444")
        self.loss_card.pack(side="right", fill="both", expand=True, padx=(8, 0))

    def setup_config_tab(self):
        # Two columns structure inside a scrollable window to cleanly arrange entries
        canvas = tk.Canvas(self.config_frame, bg="#0F172A", highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.config_frame, orient="vertical", command=canvas.yview)
        scroll_content = tk.Frame(canvas, bg="#0F172A")
        
        scroll_content.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scroll_content, anchor="nw", width=950)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        left_pane = tk.Frame(scroll_content, bg="#0F172A")
        left_pane.grid(row=0, column=0, sticky="n", padx=(0, 15))
        
        right_pane = tk.Frame(scroll_content, bg="#0F172A")
        right_pane.grid(row=0, column=1, sticky="n", padx=(15, 0))
        
        # --- LEFT PANEL: Dataset & Model Settings ---
        tk.Label(left_pane, text="DATASET & MODEL CONFIGURATION", fg="#3B82F6", bg="#0F172A", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 10))
        
        # CSV Browse field wrapper
        csv_lbl = tk.Label(left_pane, text="Local Dataset CSV Path (Blank = Pima Diabetes):", fg="#94A3B8", bg="#0F172A", font=("Segoe UI", 9, "bold"), anchor="w")
        csv_lbl.pack(fill="x", anchor="w", pady=(5, 0))
        
        csv_row = tk.Frame(left_pane, bg="#0F172A")
        csv_row.pack(fill="x", pady=2)
        
        border_frame = tk.Frame(csv_row, bg="#334155", bd=1)
        border_frame.pack(side="left", fill="x", expand=True)
        
        self.dataset_entry = tk.Entry(border_frame, bg="#1E293B", fg="#F8FAFC", insertbackground="#F8FAFC", relief="flat", bd=4, font=("Segoe UI", 10))
        self.dataset_entry.pack(fill="x")
        
        browse_btn = PremiumButton(csv_row, text="Browse", command=self.browse_dataset_file, bg="#475569", activebg="#64748B")
        browse_btn.pack(side="right", padx=(5, 0))
        
        self.target_entry = CustomEntry(left_pane, "Target Column Index (-1 for last column):", "-1")
        self.target_entry.pack(fill="x")
        
        self.impute_entry = CustomEntry(left_pane, "Missing Values Imputation Columns (Comma-separated index):", "1,2,3,4,5")
        self.impute_entry.pack(fill="x")
        
        self.batch_entry = CustomEntry(left_pane, "Mini-batch Size:", "32")
        self.batch_entry.pack(fill="x")
        
        self.hidden_entry = CustomEntry(left_pane, "Model Hidden Layers Dimensions (e.g. 16,8):", "16,8")
        self.hidden_entry.pack(fill="x")
        
        self.dropout_entry = CustomEntry(left_pane, "Model Dropout Rate:", "0.0")
        self.dropout_entry.pack(fill="x")

        # --- RIGHT PANEL: Federated & Privacy Settings ---
        tk.Label(right_pane, text="FEDERATED & SECURITY SETTINGS", fg="#3B82F6", bg="#0F172A", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 10))
        
        self.addr_entry = CustomEntry(right_pane, "Hub Coordination Address (Host:Port):", "silospan.sabyasacheemishra.com:8080")
        self.addr_entry.pack(fill="x")
        
        self.key_entry = CustomEntry(right_pane, "Client API Access Key:", "silospan_client_secret_key_2026", show="*")
        self.key_entry.pack(fill="x")
        
        self.part_entry = CustomEntry(right_pane, "Local Client Partition ID:", "0")
        self.part_entry.pack(fill="x")
        
        self.tot_entry = CustomEntry(right_pane, "Total Client Partitions:", "2")
        self.tot_entry.pack(fill="x")
        
        self.epochs_entry = CustomEntry(right_pane, "Local Epochs per Round:", "1")
        self.epochs_entry.pack(fill="x")
        
        self.lr_entry = CustomEntry(right_pane, "Local Learning Rate:", "0.01")
        self.lr_entry.pack(fill="x")
        
        # Device target target dropdown
        dev_lbl = tk.Label(right_pane, text="Computation Target:", fg="#94A3B8", bg="#0F172A", font=("Segoe UI", 9, "bold"), anchor="w")
        dev_lbl.pack(fill="x", anchor="w", pady=(5, 0))
        
        border_frame2 = tk.Frame(right_pane, bg="#334155", bd=1)
        border_frame2.pack(fill="x", pady=2)
        
        self.device_var = tk.StringVar(value="cpu")
        self.device_dropdown = ttk.Combobox(
            border_frame2, textvariable=self.device_var, values=["cpu", "cuda"], 
            state="readonly", font=("Segoe UI", 10)
        )
        self.device_dropdown.pack(fill="x")
        
        self.ssl_entry = CustomEntry(right_pane, "SSL Root Certificate CA Path (Optional):", "certs/ca.crt")
        self.ssl_entry.pack(fill="x")
        
        # Differential Privacy (LDP) Frame
        dp_section_frame = tk.Frame(right_pane, bg="#1E293B", pady=10, padx=12, highlightbackground="#334155", highlightthickness=1)
        dp_section_frame.pack(fill="x", pady=(15, 0))
        
        self.dp_enabled = tk.BooleanVar(value=True)
        dp_checkbox = tk.Checkbutton(
            dp_section_frame, text="Enable Local Differential Privacy (LDP)", 
            variable=self.dp_enabled, command=self.toggle_dp_inputs,
            bg="#1E293B", fg="#F8FAFC", selectcolor="#0F172A", 
            activebackground="#1E293B", activeforeground="#F8FAFC",
            font=("Segoe UI", 9, "bold")
        )
        dp_checkbox.pack(anchor="w")
        
        self.sigma_entry = CustomEntry(dp_section_frame, "LDP Noise Multiplier (Sigma):", "0.01")
        self.sigma_entry.pack(fill="x")
        
        self.clip_entry = CustomEntry(dp_section_frame, "LDP Weight Clipping Norm:", "1.0")
        self.clip_entry.pack(fill="x")
        
        # Synchronize DP inputs state on launch
        self.toggle_dp_inputs()

    def setup_logs_tab(self):
        # Action bar at top
        action_bar = tk.Frame(self.logs_frame, bg="#0F172A", pady=5)
        action_bar.pack(fill="x")
        
        tk.Label(action_bar, text="Console Output Logs", fg="#94A3B8", bg="#0F172A", font=("Segoe UI", 9, "bold")).pack(side="left", pady=4)
        
        # Search panel
        search_frame = tk.Frame(action_bar, bg="#0F172A")
        search_frame.pack(side="right", padx=(10, 0))
        
        border_frame = tk.Frame(search_frame, bg="#334155", bd=1)
        border_frame.pack(side="left")
        
        self.search_entry = tk.Entry(border_frame, bg="#1E293B", fg="#F8FAFC", insertbackground="#F8FAFC", relief="flat", bd=3, width=20, font=("Segoe UI", 9))
        self.search_entry.pack()
        self.search_entry.bind("<KeyRelease>", self.filter_logs)
        
        search_btn = PremiumButton(search_frame, text="Filter", command=self.filter_logs, bg="#475569", activebg="#64748B")
        search_btn.pack(side="right", padx=(5, 0))
        
        clear_btn = PremiumButton(action_bar, text="Clear Logs", command=self.clear_logs, bg="#EF4444", activebg="#DC2626")
        clear_btn.pack(side="right", padx=10)
        
        # Text panel
        text_container = tk.Frame(self.logs_frame, bg="#0d0d0d", bd=0)
        text_container.pack(fill="both", expand=True, pady=(5, 0))
        
        self.log_text = tk.Text(text_container, bg="#0d0d0d", fg="#A6E22E", font=("Consolas", 9), wrap="word", bd=0)
        self.log_text.pack(side="left", fill="both", expand=True)
        
        scrollbar = ttk.Scrollbar(text_container, orient="vertical", command=self.log_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def toggle_dp_inputs(self):
        state = "normal" if self.dp_enabled.get() else "disabled"
        self.sigma_entry.set_state(state)
        self.clip_entry.set_state(state)

    def browse_dataset_file(self):
        filename = filedialog.askopenfilename(
            title="Select Tabular Dataset CSV",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if filename:
            self.dataset_entry.delete(0, tk.END)
            self.dataset_entry.insert(0, filename)

    def clear_logs(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state="disabled")

    def filter_logs(self, event=None):
        search_val = self.search_entry.get().strip().lower()
        # Custom matching logic using highlights
        self.log_text.tag_remove("highlight", "1.0", tk.END)
        if not search_val:
            return
            
        start = "1.0"
        while True:
            pos = self.log_text.search(search_val, start, stopindex=tk.END, nocase=True)
            if not pos:
                break
            end_pos = f"{pos} + {len(search_val)} chars"
            self.log_text.tag_add("highlight", pos, end_pos)
            start = end_pos
            
        self.log_text.tag_config("highlight", background="#F59E0B", foreground="#0F172A")

    def start_training(self):
        # Read and validate all inputs from form
        try:
            addr = self.addr_entry.get().strip()
            key = self.key_entry.get().strip()
            part = int(self.part_entry.get().strip())
            tot = int(self.tot_entry.get().strip())
            epochs = int(self.epochs_entry.get().strip())
            lr = float(self.lr_entry.get().strip())
            batch = int(self.batch_entry.get().strip())
            target = int(self.target_entry.get().strip())
            dropout = float(self.dropout_entry.get().strip())
            
            # Hidden dimensions
            hidden = [int(x.strip()) for x in self.hidden_entry.get().split(",") if x.strip()]
            if not hidden:
                raise ValueError("Hidden dimensions must not be empty.")
                
            # DP parameters
            if self.dp_enabled.get():
                sigma = float(self.sigma_entry.get().strip())
                clip = float(self.clip_entry.get().strip())
                if sigma < 0 or clip <= 0:
                    raise ValueError("Differential privacy sigma must be >=0 and clipping must be >0.")
            else:
                sigma = 0.0
                clip = 1.0
                
        except ValueError as e:
            messagebox.showerror("Invalid Parameters", f"Input validation failed:\n{e}")
            return
            
        if not addr:
            messagebox.showerror("Validation Error", "Hub address cannot be blank.")
            return

        dataset_path = self.dataset_entry.get().strip()
        ssl_ca = self.ssl_entry.get().strip()
        device = self.device_var.get()
        
        impute_cols = [int(x.strip()) for x in self.impute_entry.get().split(",") if x.strip()]

        # Clean history metrics
        self.loss_history.clear()
        self.accuracy_history.clear()
        self.metrics_chart.draw_empty_chart()
        self.acc_card.set_value("0.00%")
        self.loss_card.set_value("0.0000")
        
        # Toggle buttons state
        self.start_btn.configure(state="disabled", text="RUNNING FL SESSION...", bg="#475569")
        self.stop_btn.configure(state="normal", bg="#EF4444")
        self.status_lbl.configure(text="STATUS: CONNECTING", fg="#3B82F6")
        self.status_led.set_color("#3B82F6")
        
        # Set telemetry texts
        self.telemetry_rows["server"].configure(text=addr)
        self.telemetry_rows["partition"].configure(text=f"Partition {part} of {tot}")
        self.telemetry_rows["device"].configure(text=device.upper())
        dp_text = f"Enabled (sigma={sigma})" if sigma > 0 else "Disabled"
        self.telemetry_rows["dp"].configure(text=dp_text)
        self.telemetry_rows["round_info"].configure(text="Starting...")
        
        # Clear logs and switch focus to Dashboard
        self.clear_logs()
        self.navigation.select_tab(0)
        
        # Reset and run thread
        self.stop_event.clear()
        
        self.client_thread = threading.Thread(
            target=self.run_client_worker,
            args=(addr, part, tot, epochs, lr, sigma, clip, device, ssl_ca, key, dataset_path, target, batch, impute_cols, hidden, dropout),
            daemon=True
        )
        self.client_thread.start()

    def run_client_worker(self, addr, part, tot, epochs, lr, sigma, clip, device, ssl_ca, key, dataset_path, target, batch, impute_cols, hidden, dropout):
        # Client callback function to pipe events into the queue
        def client_callback(event_type, data):
            self.queue.put((event_type, data))
            
        try:
            client.start_client(
                server_address=addr,
                partition=part,
                total_partitions=tot,
                epochs=epochs,
                lr=lr,
                dp_sigma=sigma,
                dp_clipping=clip,
                device=device,
                ssl_ca=ssl_ca,
                api_key=key,
                dataset_path=dataset_path,
                target_col=target,
                batch_size=batch,
                impute_cols=impute_cols,
                hidden_dims=hidden,
                dropout=dropout,
                callback=client_callback,
                stop_event=self.stop_event
            )
        except Exception as e:
            self.queue.put(("error", str(e)))
        finally:
            self.queue.put(("finished", {}))

    def stop_training(self):
        if messagebox.askyesno("Stop Client", "Are you sure you want to stop participating and disconnect from the Federated Learning Hub?"):
            self.stop_event.set()
            self.status_lbl.configure(text="STATUS: STOPPING...", fg="#F59E0B")
            self.status_led.set_color("#F59E0B")
            print("\n[SYSTEM] Termination signal sent. Awaiting current step completion...")

    def reset_ui_state(self):
        self.start_btn.configure(state="normal", text="START PARTICIPATING", bg="#10B981")
        self.stop_btn.configure(state="disabled", bg="#334155")
        self.status_lbl.configure(text="STATUS: IDLE", fg="#94A3B8")
        self.status_led.set_color("#64748B")
        self.epoch_progress_bar.set_progress(0.0)
        self.epoch_prog_lbl.configure(text="Epoch Progress: 0%")
        self.round_prog_lbl.configure(text="Training Status: Idle")

    def poll_queue(self):
        # Read all pending events in the queue to update GUI elements safely
        while True:
            try:
                event_type, data = self.queue.get_nowait()
            except queue.Empty:
                break
                
            if event_type == "log":
                self.log_text.configure(state="normal")
                self.log_text.insert("end", data)
                self.log_text.see("end")
                self.log_text.configure(state="disabled")
                
            elif event_type == "status":
                status = data.get("status", "idle")
                msg = data.get("message", "")
                
                if status == "connecting":
                    self.status_lbl.configure(text="STATUS: CONNECTING", fg="#3B82F6")
                    self.status_led.set_color("#3B82F6")
                    self.round_prog_lbl.configure(text=msg)
                elif status == "training":
                    self.status_lbl.configure(text="STATUS: TRAINING", fg="#F59E0B")
                    self.status_led.set_color("#F59E0B")
                    self.round_prog_lbl.configure(text=msg)
                elif status == "evaluating":
                    self.status_lbl.configure(text="STATUS: EVALUATING", fg="#10B981")
                    self.status_led.set_color("#10B981")
                    self.round_prog_lbl.configure(text=msg)
                elif status == "error":
                    self.status_lbl.configure(text="STATUS: ERROR", fg="#EF4444")
                    self.status_led.set_color("#EF4444")
                    self.round_prog_lbl.configure(text=f"Error: {msg}")
                    messagebox.showerror("Client Error", msg)
                else:
                    self.status_lbl.configure(text="STATUS: IDLE", fg="#94A3B8")
                    self.status_led.set_color("#64748B")
                    self.round_prog_lbl.configure(text=msg)
                    
            elif event_type == "round_start":
                rnd = data.get("round", 0)
                msg = data.get("message", "")
                self.telemetry_rows["round_info"].configure(text=f"Round {rnd}")
                self.status_lbl.configure(text="STATUS: TRAINING", fg="#F59E0B")
                self.status_led.set_color("#F59E0B")
                self.round_prog_lbl.configure(text=msg)
                
            elif event_type == "epoch_end":
                epoch = data.get("epoch", 1)
                max_eps = data.get("max_epochs", 1)
                loss = data.get("loss", 0.0)
                acc = data.get("accuracy", 0.0)
                
                percent = float(epoch) / float(max_eps)
                self.epoch_progress_bar.set_progress(percent)
                self.epoch_prog_lbl.configure(text=f"Epoch Progress: {epoch}/{max_eps} ({int(percent*100)}%)")
                
                # Update metrics history and redraw plot
                self.loss_history.append(loss)
                self.accuracy_history.append(acc)
                self.metrics_chart.update_metrics(self.loss_history, self.accuracy_history)
                
                # Card stats update
                self.acc_card.set_value(f"{acc*100:.2f}%")
                self.loss_card.set_value(f"{loss:.4f}")
                
            elif event_type == "eval_end":
                rnd = data.get("round", 0)
                loss = data.get("loss", 0.0)
                acc = data.get("accuracy", 0.0)
                
                # Card stats update on evaluation complete
                self.acc_card.set_value(f"{acc*100:.2f}%")
                self.loss_card.set_value(f"{loss:.4f}")
                print(f"[SYSTEM] Round {rnd} evaluation finished. Accuracy: {acc*100:.2f}% | Loss: {loss:.4f}")
                
            elif event_type == "error":
                messagebox.showerror("Execution Error", data)
                
            elif event_type == "finished":
                self.reset_ui_state()
                
            self.queue.task_done()
            
        # Re-trigger poll queue loop
        self.root.after(100, self.poll_queue)

    def cleanup(self):
        # Restore original system print streams on close
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SiloSpanClientGUI(root)
    # Handle window close cleanly
    root.protocol("WM_DELETE_WINDOW", app.cleanup)
    root.mainloop()
