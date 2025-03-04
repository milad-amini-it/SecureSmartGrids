import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import hashlib
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import SklearnClassifier
import flwr as fl
import pandapower as pp
import pandapower.networks as pn
import warnings
import sys

# Suppress optional PyTorch warnings from ART
warnings.filterwarnings("ignore", message="PyTorch not found")

# Load IEEE 14-Bus system from .raw file
try:
    # Convert .raw to pandapower network (assuming IEEE 14 bus.raw is in the directory)
    net = pp.from_pypower("IEEE 14 bus.raw", fformat="raw")
    print("IEEE 14-Bus system loaded successfully.")
except FileNotFoundError:
    print("Please ensure 'IEEE 14 bus.raw' is in the working directory.")
    sys.exit(1)

# Simulate operational data (e.g., bus voltages) over 5000 time steps
n_samples = 5000
np.random.seed(42)
X = np.zeros((n_samples, len(net.bus)))  # One feature per bus (voltage magnitude)
for i in range(n_samples):
    pp.runpp(net)  # Run power flow
    X[i] = net.res_bus.vm_pu.values  # Voltage magnitude in per unit
    # Add small random variation to simulate time series
    net.load.p_mw += np.random.normal(0, 0.01, len(net.load))
    net.gen.p_mw += np.random.normal(0, 0.01, len(net.gen))

# Preprocess data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Simulate attacks: Label 20% as anomalies and perturb them
y = np.zeros(n_samples, dtype=int)
attack_idx = np.random.choice(n_samples, size=int(0.2 * n_samples), replace=False)
y[attack_idx] = 1
X[attack_idx] += np.random.normal(0, 0.5, X[attack_idx].shape)  # Add noise for attacks

# Split into train and test
train_size = int(0.8 * n_samples)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Simulated blockchain ledger for model updates
blockchain_ledger = []

def hash_update(update):
    """Simulate blockchain hashing of model updates (e.g., predictions)."""
    return hashlib.sha256(str(update).encode()).hexdigest()

def log_to_blockchain(update):
    """Log model update hash to a simulated blockchain ledger."""
    hashed_update = hash_update(update)
    blockchain_ledger.append(hashed_update)
    return hashed_update

def adversarial_training(local_X, local_y):
    """Perform adversarial training using scikit-learn SVM and ART."""
    classifier = SVC(kernel='linear', C=1.0, probability=True)
    classifier.fit(local_X, local_y)

    art_classifier = SklearnClassifier(model=classifier)
    attack = FastGradientMethod(estimator=art_classifier, eps=0.1)
    X_adv = attack.generate(x=local_X)

    X_combined = np.vstack((local_X, X_adv))
    y_combined = np.hstack((local_y, local_y))
    classifier.fit(X_combined, y_combined)
    return classifier

# Flower client
class SmartGridClient(fl.client.NumPyClient):
    """Flower client simulating an edge device in the smart grid."""
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.model = None

    def fit(self, parameters, config):
        """Train the local model with adversarial examples."""
        self.model = adversarial_training(self.X, self.y)
        y_prob = self.model.predict_proba(X_test)[:, 1]  # Probabilities for aggregation
        log_to_blockchain(y_prob.tolist())
        return [], len(self.X), {"y_prob": y_prob.tolist()}  # Empty params, send probabilities

    def evaluate(self, parameters, config):
        """Evaluate the local model on test data."""
        y_pred = self.model.predict(X_test)
        accuracy = f1_score(y_test, y_pred)
        return float(accuracy), len(X_test), {"f1_score": float(accuracy)}

def client_fn(cid):
    """Create a Flower client for each edge device."""
    start_idx = int(cid) * 1000  # Split 4000 train samples among 4 clients
    end_idx = start_idx + 1000
    return SmartGridClient(X_train[start_idx:end_idx], y_train[start_idx:end_idx])

# Flower server strategy
class SmartGridStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        """Aggregate client predictions instead of parameters."""
        if not results:
            return None, {}
        client_probs = [fl.common.parameters_to_ndarrays(metrics["y_prob"])[0] for _, metrics in results]
        aggregated_probs = np.mean(client_probs, axis=0)
        log_to_blockchain(aggregated_probs.tolist())
        return [], {"aggregated_probs": aggregated_probs.tolist()}

    def aggregate_evaluate(self, server_round, results, failures):
        """Evaluate aggregated predictions."""
        if not results:
            return None, {}
        accuracies = [r.metrics["f1_score"] for _, r in results]
        return float(np.mean(accuracies)), {"avg_f1_score": float(np.mean(accuracies))}

def run_server():
    """Start the Flower server."""
    print("Starting Flower Server on port 8081...")
    strategy = SmartGridStrategy(
        min_fit_clients=4,
        min_evaluate_clients=4,
        min_available_clients=4,
    )
    try:
        fl.server.start_server(
            server_address="localhost:8081",
            config=fl.server.ServerConfig(num_rounds=3),
            strategy=strategy,
        )
    except Exception as e:
        print(f"Server failed: {e}. Try changing the port (e.g., 8082).")

def run_client(cid):
    """Start a Flower client."""
    print(f"Starting Client {cid}...")
    fl.client.start_numpy_client(
        server_address="localhost:8081",
        client=client_fn(cid),
    )

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "client":
        if len(sys.argv) > 2:
            run_client(sys.argv[2])
        else:
            print("Please provide a client ID (e.g., 'python smart_grid_security.py client 0')")
    else:
        run_server()