"""FidelityFloor: Isaac Sim as a degradable oracle for world-model fidelity requirements.

Launcher-agnostic core. Nothing in this package imports Modal; Isaac Sim is imported
lazily inside envs.py/oracle.py so every other module runs on a CPU-only box.
"""

__version__ = "0.1.0"
