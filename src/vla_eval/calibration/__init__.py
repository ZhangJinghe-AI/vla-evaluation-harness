"""Calibration dataset export for downstream quantizers (e.g. QVLA).

Layout::

    format.py   — shared version-2 ``.npz`` I/O (any camera layout)
    sampling.py — shared chunk-grid / phase quota helpers
    rollout.py  — FP eval rollouts → calib (all benchmarks)
    libero.py   — legacy LIBERO demo HDF5 replay → calib
"""

from vla_eval.calibration.format import CALIBRATION_FILE_VERSION, save_calibration_dataset
from vla_eval.calibration.libero import export_libero_dataset
from vla_eval.calibration.rollout import RolloutRecorder, export_rollout_calibration

__all__ = [
    "CALIBRATION_FILE_VERSION",
    "RolloutRecorder",
    "export_libero_dataset",
    "export_rollout_calibration",
    "save_calibration_dataset",
]
