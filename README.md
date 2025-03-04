# SecureSmartGrids
# Smart Grid Security Framework

This repository implements a security framework for edge-based smart grids, integrating adversarial machine learning (scikit-learn + ART), federated learning (Flower), and blockchain (simulated with hashlib).

## Requirements
Install dependencies:
```bash
pip install -r requirements.txt

```
Usage

Run the server:
 ```bash

    python smart_grid_security.py
```
Run 4 Clients in Separate Terminals
```bash

    python smart_grid_security.py client 0
    python smart_grid_security.py client 1
    python smart_grid_security.py client 2
    python smart_grid_security.py client 3
```

Output:

Server will show aggregated results after 3 rounds.
Clients will report local F1-scores and latencies.
Final accuracy will depend on the data.

you can run this project with simulated data (insted of client server) and ElectricityLoadDiagrams20112014 from UCI Machine Learning Repository for easier Run project
make sure file LD2011_2014.txt exist on running path then Run
 ```bash
    python smart_grid_security_with_simulated_data.py
```
Features

Adversarial training with SVM and FGSM.
Federated learning across 4 simulated edge devices.
Blockchain logging of model updates.

Paper Reference

Based on: "Securing Edge-Based Smart Grids with Privacy-Preserving Adversarial Machine Learning and Blockchain Integration."

