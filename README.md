# SiloSpan Edge Client: Operator Guide

This guide is designed for **Edge Silo Administrators** running client nodes to participate in SiloSpan federated learning rounds.

As a client operator, **your raw data never leaves your infrastructure**. Your node localizes all database queries and model training, applying Local Differential Privacy (LDP) to secure parameters before returning updates to the central hub.

---

## 1. System Requirements & Setup

### Requirements
*   **Operating System**: Linux (Ubuntu/Debian recommended), macOS, or Windows 10/11.
*   **Python**: Version 3.10 or 3.11.
*   **Hardware**: CPU is sufficient, but CUDA-compatible GPU is supported for faster training execution.

---

### Step-by-Step Installation

1. **Clone or Download the client files**:
   Ensure you have `client.py`, the `model/` folder, and `requirements.txt` in your execution folder.

2. **Create a Virtual Environment**:
   ```bash
   # Create environment
   python -m venv .venv

   # Activate environment (Windows)
   .venv\Scripts\activate

   # Activate environment (Linux/macOS)
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 2. Running the Client Node

Start the client using `client.py` and point it to the central hub domain address.

### Basic Startup Command (With Auto-Enrollment)
```bash
python client.py silospan.sabyasacheemishra.com:8080 \
  --partition 0 \
  --ssl-ca certs/ca.crt \
  --api-key silospan_client_secret_key_2026
```

### Parameter Reference

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `server_address` | Positional | `localhost:8080` | gRPC address of the coordination hub. |
| `--partition` | Integer | `0` | Local partition ID of your dataset (e.g. `0`, `1`). |
| `--total-partitions` | Integer | `2` | Total number of client partitions. |
| `--epochs` | Integer | `1` | Number of training epochs to execute locally per round. |
| `--lr` | Float | `0.01` | Learning rate for backpropagation optimizer. |
| `--device` | String | `cpu` | Device computing target: `'cpu'` or `'cuda'`. |
| `--ssl-ca` | String | *None* | Local path where the hub's CA SSL certificate is saved. |
| `--api-key` | String | *None* | Pre-shared key to authorize CA certificate auto-enrollment. |
| `--api-url` | String | *None* | Custom URL if the enrollment endpoint is on a non-standard port. |

---

## 3. Local Data Privacy (Differential Privacy)

SiloSpan clients can automatically inject mathematical noise into weight updates before sending them back to the server to prevent leakage of individual records.

*   **Weight Clipping (`--dp-clipping`)**: Sets the maximum $L_2$ norm sensitivity boundary of the model weight updates. Default is `1.0`.
*   **Noise Multiplier (`--dp-sigma`)**: Determines the scale of Gaussian noise added to the clipped parameters. Default is `0.0` (disabled). Set to `0.01` or `0.05` to enable differential privacy.

#### Run with Local Privacy Enabled:
```bash
python client.py silospan.sabyasacheemishra.com:8080 \
  --partition 0 \
  --ssl-ca certs/ca.crt \
  --api-key silospan_client_secret_key_2026 \
  --dp-sigma 0.02 \
  --dp-clipping 1.0
```

---

## 4. Running as a Desktop GUI Application (No Terminal Required!)

For non-technical operators or administrators, a graphical interface can be used to set parameters, select files, and view logs.

### Running via Python
Activate your virtual environment and run:
```bash
python client_gui.py
```
This opens a dark-themed control window containing inputs for your Hub address, credentials, dataset settings, and a real-time console log terminal.

### Packaging into a Standalone Executable (.exe)
You can compile this GUI script into a single, double-clickable executable file that runs on other machines without needing Python installed.

1. Install PyInstaller inside your virtual environment:
   ```bash
   pip install pyinstaller
   ```
2. Build the executable:
   ```bash
   pyinstaller --noconsole --onefile client_gui.py
   ```
3. Once completed, find the single `client_gui.exe` (or binary equivalent) inside the newly created `dist/` folder. You can distribute this executable directly to your client operators!

---

## 5. Running persistently in the Background (Daemon)

To ensure the client runs continuously, restarts if it crashes, and boots on system start, deploy it as a service daemon:

### Linux Daemon (Systemd)
1. Write a systemd service file at `/etc/systemd/system/silospan-client.service`:
   ```ini
   [Unit]
   Description=SiloSpan Edge Client Daemon
   After=network.target

   [Service]
   Type=simple
   User=ubuntu
   WorkingDirectory=/home/ubuntu/silospan
   ExecStart=/home/ubuntu/silospan/.venv/bin/python client.py silospan.sabyasacheemishra.com:8080 --partition 0 --ssl-ca certs/ca.crt --api-key silospan_client_secret_key_2026
   Restart=always
   RestartSec=5
   Environment=PYTHONUNBUFFERED=1

   [Install]
   WantedBy=multi-user.target
   ```
2. Enable and start the daemon:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable silospan-client.service
   sudo systemctl start silospan-client.service
   ```

### Windows Daemon (NSSM)
1. Download **NSSM (Non-Sucking Service Manager)** from `https://nssm.cc`.
2. Run the GUI installer in Administrator PowerShell:
   ```powershell
   .\nssm.exe install SiloSpanClient
   ```
3. Set **Path** to your virtual environment python (e.g. `C:\silospan\.venv\Scripts\python.exe`).
4. Set **Arguments** to:
   `client.py silospan.sabyasacheemishra.com:8080 --partition 0 --ssl-ca certs/ca.crt --api-key silospan_client_secret_key_2026`
5. Click **Install service** and start it using:
   ```powershell
   Start-Service SiloSpanClient
   ```

---

## 5. Operations Telemetry & Troubleshooting

### Reconnection Behavior
The client includes built-in exponential backoff retry logic. If the hub is offline or your local internet connection drops:
*   The client will wait **2 seconds** and attempt reconnection.
*   The delay doubles on subsequent failures (`4.0s`, `8.0s`, `16.0s`, `32.0s`) up to a maximum of **60 seconds**.
*   It will retry indefinitely without crashing, automatically reconnecting once the hub or internet is restored.

### Exception Logs
If configured by your system administrator, you can export your **Sentry DSN** to send local stack traces directly to the developer console for debugging:
```bash
# Linux
export SENTRY_DSN="your_sentry_dsn_here"

# Windows PowerShell
$env:SENTRY_DSN="your_sentry_dsn_here"
```
No private dataset content is sent to Sentry; only runtime exceptions (e.g. "GPU out of memory" or "Dataset not found") are logged.
