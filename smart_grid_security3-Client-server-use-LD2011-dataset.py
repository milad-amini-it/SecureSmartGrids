import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import hashlib
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import SklearnClassifier
import flwr as fl
import pandas as pd
import warnings
import sys

# Suppress optional PyTorch warnings from ART
warnings.filterwarnings("ignore", message="PyTorch not found")

# Load real smart grid data (ElectricityLoadDiagrams20112014 from UCI)
# Download from: https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014
try:
    data = pd.read_csv("LD2011_2014.txt", sep=";", decimal=",", index_col=0)
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Please download 'LD2011_2014.txt' from UCI and place it in the working directory.")
    sys.exit(1)

# Preprocess data: Use a subset and create binary labels (normal vs. anomaly)
data = data.iloc[:5000, :10]  # Use first 5000 rows and 10 clients’ data as features
X = data.values
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Simulate attacks: Randomly label 20% as anomalies and perturb them
np.random.seed(42)
y = np.zeros(len(X), dtype=int)
attack_idx = np.random.choice(len(X), size=int(0.2 * len(X)), replace=False)
y[attack_idx] = 1
X[attack_idx] += np.random.normal(0, 0.5, X[attack_idx].shape)  # Add noise to simulate attacks

# Split into train and test
train_size = int(0.8 * len(X))
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
        return [], len(self.X), {"y_prob": y_prob.tolist()}  # Return empty params, send probabilities

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
        # Collect probabilities from all clients
        client_probs = [fl.common.parameters_to_ndarrays(metrics["y_prob"])[0] for _, metrics in results]
        aggregated_probs = np.mean(client_probs, axis=0)
        log_to_blockchain(aggregated_probs.tolist())
        # Return empty parameters since we're aggregating predictions
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