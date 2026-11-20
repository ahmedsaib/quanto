"""Submission interface on four training-fitted PCA features, without IDs."""
from .models import QuantumKernelClassifier


class Submission(QuantumKernelClassifier):
    """4-qubit RY Fidelity Quantum Kernel + Classical SVM (C=0.25)."""
    def __init__(self):
        super().__init__(labels=('alfalfa', 'maize', 'wheat'), C=0.25)