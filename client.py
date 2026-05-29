import flwr as fl
import sys
import argparse
import numpy as np
import os
import sentry_sdk
from collections import OrderedDict
from model import SiloSpanClassifier, load_diabetes_data
from model.utils import load_tabular_data

# Custom exception for user interruption
class InterruptedError(Exception):
    pass

# Initialize Sentry if configured
SENTRY_DSN = os.getenv("SENTRY_DSN")
if SENTRY_DSN:
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        traces_sample_rate=1.0,
        profiles_sample_rate=1.0,
    )
    print("[SILOSPAN CLIENT] Sentry SDK initialized.")

def compute_loss(logits, y):
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    n = logits.shape[0]
    loss = -np.sum(np.log(probs[np.arange(n), y] + 1e-15)) / n
    return loss

def run_local_backpropagation(model, train_loader, epochs=1, lr=0.01, device="cpu", callback=None, stop_event=None, partition_id=0):
    """
    Performs standard backpropagation over local client dataset partition using NumPy.
    """
    model.train()
    
    for epoch in range(epochs):
        if stop_event and stop_event.is_set():
            raise InterruptedError("Training interrupted by operator.")
            
        running_loss = 0.0
        correct = 0
        total = 0
        for data, target in train_loader:
            if stop_event and stop_event.is_set():
                raise InterruptedError("Training interrupted by operator.")
            
            output = model.forward(data)
            loss = compute_loss(output, target)
            model.backward(data, target, lr=lr)
            
            running_loss += loss * len(data)
            pred = output.argmax(axis=1)
            correct += np.sum(pred == target)
            total += len(data)
        
        epoch_loss = running_loss / total if total > 0 else 0.0
        epoch_acc = correct / total if total > 0 else 0.0
        
        print(f"[CLIENT {partition_id}] Epoch {epoch+1}/{epochs} - Training Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc * 100:.2f}%")
        if callback:
            callback("epoch_end", {
                "epoch": epoch + 1,
                "max_epochs": epochs,
                "loss": epoch_loss,
                "accuracy": epoch_acc
            })

def run_local_evaluation(model, test_loader, device="cpu"):
    """
    Evaluates the model on local validation/test sets to report local performance metrics using NumPy.
    """
    model.eval()
    loss = 0.0
    correct = 0
    total = 0
    for data, target in test_loader:
        output = model.forward(data)
        loss += compute_loss(output, target) * len(data)
        pred = output.argmax(axis=1)
        correct += np.sum(pred == target)
        total += len(data)
            
    avg_loss = loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, total, accuracy

class SiloSpanClient(fl.client.NumPyClient):
    def __init__(self, partition_id, train_loader, test_loader, epochs, lr, dp_sigma, dp_clipping, device,
                 input_dim=8, hidden_dims=None, dropout=0.0, callback=None, stop_event=None):
        if hidden_dims is None:
            hidden_dims = [16, 8]
        self.model = SiloSpanClassifier(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
        self.partition_id = partition_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.lr = lr
        self.dp_sigma = dp_sigma
        self.dp_clipping = dp_clipping
        self.device = device
        self.callback = callback
        self.stop_event = stop_event
        self.round_counter = 0

    def get_parameters(self, config):
        params = []
        for w, b in zip(self.model.weights, self.model.biases):
            params.append(w)
            params.append(b)
        return params

    def set_parameters(self, parameters):
        idx = 0
        for i in range(len(self.model.weights)):
            self.model.weights[i] = parameters[idx]
            self.model.biases[i] = parameters[idx+1]
            idx += 2

    def fit(self, parameters, config):
        self.round_counter += 1
        if self.stop_event and self.stop_event.is_set():
            raise InterruptedError("Fit process interrupted by user.")
            
        if self.callback:
            self.callback("round_start", {
                "round": self.round_counter,
                "status": "training",
                "message": f"Round {self.round_counter}: Training started..."
            })
            
        old_params = parameters
        self.set_parameters(parameters)
        
        # Run local training
        run_local_backpropagation(
            self.model, 
            self.train_loader, 
            epochs=self.epochs, 
            lr=self.lr, 
            device=self.device,
            callback=self.callback,
            stop_event=self.stop_event,
            partition_id=self.partition_id
        )
        
        new_params = self.get_parameters(config={})
        
        # Apply Local Differential Privacy (DP)
        if self.dp_sigma > 0:
            print(f"[CLIENT {self.partition_id}] Applying Differential Privacy (sigma={self.dp_sigma}, clip={self.dp_clipping}) to updates...")
            perturbed_params = []
            updates = [new - old for new, old in zip(new_params, old_params)]
            
            # Clip updates to limit sensitivity
            if self.dp_clipping > 0:
                total_norm = np.sqrt(sum(np.sum(np.square(u)) for u in updates))
                clip_coef = min(1.0, self.dp_clipping / (total_norm + 1e-6))
                if clip_coef < 1.0:
                    updates = [u * clip_coef for u in updates]
            
            # Add Gaussian noise
            for u in updates:
                noise = np.random.normal(0.0, self.dp_sigma * self.dp_clipping, size=u.shape)
                perturbed_update = u + noise
                perturbed_params.append(perturbed_update)
                
            # Reconstruct model parameters: old_params + perturbed_update
            new_params = [old + pert for old, pert in zip(old_params, perturbed_params)]
            self.set_parameters(new_params)

        num_samples = len(self.train_loader.dataset)
        
        if self.callback:
            self.callback("round_end", {
                "round": self.round_counter,
                "status": "idle",
                "message": f"Round {self.round_counter}: Training finished."
            })
            
        return new_params, num_samples, {
            "silo_id": self.partition_id,
            "dp_enabled": bool(self.dp_sigma > 0),
            "dp_sigma": float(self.dp_sigma),
            "dp_clipping": float(self.dp_clipping),
            "num_samples": int(num_samples)
        }

    def evaluate(self, parameters, config):
        if self.stop_event and self.stop_event.is_set():
            raise InterruptedError("Evaluation process interrupted by user.")
            
        if self.callback:
            self.callback("status", {
                "status": "evaluating",
                "message": f"Round {self.round_counter}: Evaluating model..."
            })
            
        self.set_parameters(parameters)
        loss, total, accuracy = run_local_evaluation(self.model, self.test_loader, device=self.device)
        print(f"[CLIENT {self.partition_id}] Local Evaluation - Loss: {loss:.4f} | Accuracy: {accuracy * 100:.2f}%")
        
        if self.callback:
            self.callback("eval_end", {
                "round": self.round_counter,
                "loss": loss,
                "accuracy": accuracy,
                "num_samples": total
            })
            
        return float(loss), int(total), {"accuracy": float(accuracy)}

def get_enrollment_url(server_address: str) -> str:
    host = server_address.split(":")[0]
    if host in ["localhost", "127.0.0.1"]:
        return "http://localhost:8000"
    return f"https://{host}"

def auto_enroll_ca(server_address: str, api_key: str, ca_path: str, api_url: str = ""):
    import urllib.request
    import json
    
    enroll_url = api_url if api_url else get_enrollment_url(server_address)
    url = f"{enroll_url.rstrip('/')}/api/auth/enroll"
    print(f"[SILOSPAN CLIENT] CA certificate auto-enrollment requested via {url}...")
    
    data = json.dumps({"api_key": api_key}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    
    try:
        import ssl
        context = ssl.create_default_context()
        if "localhost" in url or "127.0.0.1" in url:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            
        with urllib.request.urlopen(req, context=context) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            ca_content = res_data["ca_certificate"]
            
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(ca_path) or ".", exist_ok=True)
            with open(ca_path, "w") as f:
                f.write(ca_content)
            print(f"[SILOSPAN CLIENT] Auto-enrollment successful! Saved CA root certificate to '{ca_path}'")
    except Exception as e:
        print(f"[SILOSPAN CLIENT] Auto-enrollment failed: {e}")
        sys.exit(1)

def start_client(
    server_address: str = "localhost:8080",
    partition: int = 0,
    total_partitions: int = 2,
    epochs: int = 1,
    lr: float = 0.01,
    dp_sigma: float = 0.0,
    dp_clipping: float = 1.0,
    device: str = "cpu",
    ssl_ca: str = "",
    api_key: str = "",
    api_url: str = "",
    # Custom configurations
    dataset_path: str = "",
    target_col: int = -1,
    batch_size: int = 32,
    impute_cols: list = None,
    hidden_dims: list = None,
    dropout: float = 0.0,
    callback = None,
    stop_event = None
):
    # Auto-enroll CA cert if API key is provided and cert is missing
    if api_key and ssl_ca:
        if not os.path.exists(ssl_ca):
            try:
                auto_enroll_ca(server_address, api_key, ssl_ca, api_url)
            except Exception as e:
                print(f"[SILOSPAN CLIENT] CA certificate auto-enrollment failed: {e}")
                if callback:
                    callback("status", {"status": "error", "message": f"Auto-enrollment failed: {e}"})
                return

    if callback:
        callback("status", {"status": "connecting", "message": f"Connecting to Hub: {server_address}..."})
        
    print(f"[SILOSPAN CLIENT] Connecting to Hub: {server_address}")
    print(f"[SILOSPAN CLIENT] Loading local data partition {partition}/{total_partitions}...")
    
    try:
        if dataset_path and os.path.exists(dataset_path):
            print(f"[SILOSPAN CLIENT] Loading custom dataset from {dataset_path}...")
            train_loader, test_loader = load_tabular_data(
                file_path=dataset_path,
                target_col=target_col,
                partition_id=partition,
                num_partitions=total_partitions,
                batch_size=batch_size,
                impute_cols=impute_cols if impute_cols is not None else [1, 2, 3, 4, 5],
                scale=True
            )
        else:
            if dataset_path:
                print(f"[SILOSPAN CLIENT] Warning: Dataset path '{dataset_path}' not found. Falling back to default Pima dataset.")
            train_loader, test_loader = load_diabetes_data(
                partition_id=partition, 
                num_partitions=total_partitions, 
                batch_size=batch_size
            )
    except Exception as e:
        err_msg = f"Failed to load dataset: {e}"
        print(f"[SILOSPAN CLIENT] Error: {err_msg}")
        if callback:
            callback("status", {"status": "error", "message": err_msg})
        return

    # Extract input feature dimensions
    try:
        sample_features, _ = train_loader.dataset[0]
        input_dim = sample_features.shape[0]
        print(f"[SILOSPAN CLIENT] Autodetected features count: {input_dim}")
    except Exception as e:
        err_msg = f"Failed to inspect features: {e}"
        print(f"[SILOSPAN CLIENT] Error: {err_msg}")
        if callback:
            callback("status", {"status": "error", "message": err_msg})
        return

    client = SiloSpanClient(
        partition_id=partition,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=epochs,
        lr=lr,
        dp_sigma=dp_sigma,
        dp_clipping=dp_clipping,
        device=device,
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        callback=callback,
        stop_event=stop_event
    )
    
    # Load CA root certificates for secure TLS connection
    root_certificates = None
    if ssl_ca:
        try:
            from pathlib import Path
            print("[SILOSPAN CLIENT] Establishing secure gRPC TLS channel using Root CA cert...")
            root_certificates = Path(ssl_ca).read_bytes()
        except Exception as e:
            print(f"[SILOSPAN CLIENT] Warning: Failed to load CA certificate: {e}. Falling back to insecure connection.")

    import time
    retry_delay = 2.0
    max_delay = 60.0
    backoff_factor = 2.0
    
    while True:
        if stop_event and stop_event.is_set():
            print("[SILOSPAN CLIENT] Stop requested before connection attempt.")
            if callback:
                callback("status", {"status": "idle", "message": "Stopped by operator."})
            break
            
        try:
            if callback:
                callback("status", {"status": "connecting", "message": f"Connecting to Hub at {server_address}..."})
            print(f"[SILOSPAN CLIENT] Connecting to Hub at {server_address}...")
            
            fl.client.start_client(
                server_address=server_address,
                client=client.to_client(),
                root_certificates=root_certificates
            )
            print("[SILOSPAN CLIENT] Session finished successfully.")
            if callback:
                callback("status", {"status": "idle", "message": "Session completed successfully."})
            break
        except InterruptedError as e:
            print(f"[SILOSPAN CLIENT] Interrupted: {e}")
            if callback:
                callback("status", {"status": "idle", "message": "Stopped by operator."})
            break
        except Exception as e:
            print(f"[SILOSPAN CLIENT] Connection error or disconnected: {e}")
            # Send exception to Sentry if active
            if os.getenv("SENTRY_DSN"):
                sentry_sdk.capture_exception(e)
                
            if stop_event and stop_event.is_set():
                if callback:
                    callback("status", {"status": "idle", "message": "Stopped by operator."})
                break
                
            print(f"[SILOSPAN CLIENT] Retrying connection in {retry_delay:.1f} seconds...")
            if callback:
                callback("status", {
                    "status": "connecting", 
                    "message": f"Disconnected. Retrying in {retry_delay:.1f}s..."
                })
            
            # Wait with check interval for stop_event responsiveness
            for _ in range(int(retry_delay * 2)):
                if stop_event and stop_event.is_set():
                    break
                time.sleep(0.5)
                
            if stop_event and stop_event.is_set():
                if callback:
                    callback("status", {"status": "idle", "message": "Stopped by operator."})
                break
                
            retry_delay = min(retry_delay * backoff_factor, max_delay)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SiloSpan distributed client node.")
    parser.add_argument("server_address", type=str, nargs="?", default="localhost:8080", 
                        help="Address of the centralized hub (e.g. host:port).")
    parser.add_argument("--partition", type=int, default=0, help="Local partition ID of the dataset.")
    parser.add_argument("--total-partitions", type=int, default=2, help="Total number of partitions.")
    parser.add_argument("--epochs", type=int, default=1, help="Local epochs to train in each round.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--dp-sigma", type=float, default=0.0, help="Differential privacy noise multiplier (0.0 to disable).")
    parser.add_argument("--dp-clipping", type=float, default=1.0, help="Differential privacy update clipping norm.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use ('cpu' or 'cuda').")
    parser.add_argument("--ssl-ca", type=str, default="", help="Path to CA root certificate for secure TLS.")
    parser.add_argument("--api-key", type=str, default="", help="Client authentication API key for auto-enrollment.")
    parser.add_argument("--api-url", type=str, default="", help="FastAPI portal base URL (optional).")
    
    # Advanced parameters
    parser.add_argument("--dataset-path", type=str, default="", help="Custom CSV dataset path.")
    parser.add_argument("--target-col", type=int, default=-1, help="Index of target label column.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size for training.")
    parser.add_argument("--hidden-dims", type=str, default="16,8", help="Comma-separated hidden layer dimensions.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability.")
    
    args = parser.parse_args()
    
    # Parse hidden dimensions
    hidden = [int(x.strip()) for x in args.hidden_dims.split(",") if x.strip()]
    
    start_client(
        server_address=args.server_address,
        partition=args.partition,
        total_partitions=args.total_partitions,
        epochs=args.epochs,
        lr=args.lr,
        dp_sigma=args.dp_sigma,
        dp_clipping=args.dp_clipping,
        device=args.device,
        ssl_ca=args.ssl_ca,
        api_key=args.api_key,
        api_url=args.api_url,
        dataset_path=args.dataset_path,
        target_col=args.target_col,
        batch_size=args.batch_size,
        hidden_dims=hidden,
        dropout=args.dropout
    )
