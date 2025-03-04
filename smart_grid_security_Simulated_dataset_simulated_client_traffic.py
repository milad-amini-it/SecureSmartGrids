import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import hashlib
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import SklearnClassifier
import time
import warnings

# Suppress optional PyTorch warnings from ART
warnings.filterwarnings("ignore", message="PyTorch not found")

# Simulated realistic smart grid data (with some structure)
np.random.seed(42)
def generate_smart_grid_data(n_samples, n_features=10):
    """Generate synthetic smart grid data with some structure."""
    X = np.random.rand(n_samples, n_features)
    # Add a simple pattern: attack samples (label 1) have higher values in first 3 features
    y = np.random.randint(0, 2, n_samples)
    X[y == 1, :3] += 0.5  # Increase first 3 features for attack samples
    return X, y

X_train, y_train = generate_smart_grid_data(5000)  # More samples
X_test, y_test = generate_smart_grid_data(1000)

# Simulated blockchain ledger for model updates
blockchain_ledger = []

def hash_update(update):
    """Simulate blockchain hashing of model updates (e.g., predictions or metrics)."""
    return hashlib.sha256(str(update).encode()).hexdigest()

def log_to_blockchain(update):
    """Log model update hash to a simulated blockchain ledger."""
    hashed_update = hash_update(update)
    blockchain_ledger.append(hashed_update)
    return hashed_update

def adversarial_training(local_X, local_y):
    """Perform adversarial training using scikit-learn SVM and ART."""
    # Initialize SVM classifier with optimized parameters
    classifier = SVC(kernel='linear', C=1.0, probability=True)
    classifier.fit(local_X, local_y)

    # Wrap the classifier with ART for adversarial attacks
    art_classifier = SklearnClassifier(model=classifier)

    # Generate adversarial examples using Fast Gradient Sign Method (FGSM)
    attack = FastGradientMethod(estimator=art_classifier, eps=0.1)
    X_adv = attack.generate(x=local_X)

    # Combine original and adversarial data
    X_combined = np.vstack((local_X, X_adv))
    y_combined = np.hstack((local_y, local_y))

    # Retrain the model with adversarial examples
    classifier.fit(X_combined, y_combined)
    return classifier

class SmartGridClient:
    """Simulated client for federated learning."""
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.model = None

    def fit(self):
        """Train the local model with adversarial examples."""
        start_time = time.time()
        self.model = adversarial_training(self.X, self.y)
        latency = time.time() - start_time
        # Predict probabilities on test data for aggregation
        y_prob = self.model.predict_proba(X_test)[:, 1]  # Probability of class 1
        log_to_blockchain(y_prob.tolist())  # Log probabilities to blockchain
        return y_prob, {"latency": latency}

    def evaluate(self):
        """Evaluate the local model on test data."""
        y_pred = self.model.predict(X_test)
        accuracy = f1_score(y_test, y_pred)
        return accuracy

def simulate_federated_learning():
    """Simulate federated learning with 4 clients in a single process."""
    print("Starting Simulated Federated Learning...")

    # Simulate 4 edge devices
    clients = []
    for i in range(4):
        start_idx = i * 1250  # 5000 samples / 4 clients
        end_idx = start_idx + 1250
        clients.append(SmartGridClient(X_train[start_idx:end_idx], y_train[start_idx:end_idx]))

    # Simulate 5 rounds of federated learning (increased from 3)
    global_probabilities = np.zeros(len(y_test))
    for round_num in range(1, 6):
        print(f"Round {round_num}")
        client_probabilities = []
        for i, client in enumerate(clients):
            y_prob, metrics = client.fit()
            client_probabilities.append(y_prob)
            print(f"Client {i} latency: {metrics['latency']:.4f} seconds")

        # Aggregate probabilities (average instead of majority vote)
        global_probabilities = np.mean(client_probabilities, axis=0)
        log_to_blockchain(global_probabilities.tolist())

        # Evaluate each client
        for i, client in enumerate(clients):
            accuracy = client.evaluate()
            print(f"Client {i} F1-Score: {accuracy:.4f}")

    # Final evaluation based on aggregated probabilities
    final_predictions = (global_probabilities > 0.5).astype(int)
    final_accuracy = f1_score(y_test, final_predictions)
    print(f"Final Attack Detection Accuracy: {final_accuracy * 100:.2f}%")
    print(f"Blockchain Ledger (first 2 entries): {blockchain_ledger[:2]}")

if __name__ == "__main__":
    simulate_federated_learning()