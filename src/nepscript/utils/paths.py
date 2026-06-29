"""
Results directory resolution.

Single source of truth for locating the experiment results bundle
(``NAS_Experiment_Results/``).  The bundle is large and lives on Google Drive,
so it is **not** committed to git.  Anyone who downloads it can place it
anywhere and point the code at it without editing source files.

Resolution order (first match wins):

1. An explicit path argument (e.g. a ``--results-dir`` CLI flag).
2. The ``NAS_RESULTS_DIR`` environment variable.
3. The default ``NAS_Experiment_Results`` folder in the project root.

The downloaded bundle uses its own subfolder names.  The helper functions
below map logical names used throughout the codebase to the actual folders,
so the layout is defined in exactly one place.
"""

import os
from pathlib import Path

# Project root = three levels up from this file:
# src/nepscript/utils/paths.py -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Environment variable users can set once to make the location global.
ENV_VAR = "NAS_RESULTS_DIR"

# Default folder name expected in the project root.
DEFAULT_DIRNAME = "NAS_Experiment_Results"


def get_results_dir(cli_arg: str | None = None) -> Path:
    """
    Resolve the root of the results bundle.

    Args:
        cli_arg: Path passed on the command line (e.g. ``--results-dir``).
                 Takes precedence over everything else when provided.

    Returns:
        Absolute :class:`~pathlib.Path` to the results bundle root.
        The path is returned even if it does not exist so callers can emit a
        clear error; use :func:`require_results_dir` to fail fast instead.
    """
    if cli_arg:
        return Path(cli_arg).expanduser().resolve()

    env_value = os.environ.get(ENV_VAR)
    if env_value:
        return Path(env_value).expanduser().resolve()

    return (PROJECT_ROOT / DEFAULT_DIRNAME).resolve()


def require_results_dir(cli_arg: str | None = None) -> Path:
    """
    Resolve the results bundle and verify it exists.

    Raises:
        FileNotFoundError: If the resolved directory is missing, with a message
            explaining how to point the code at the downloaded folder.
    """
    base = get_results_dir(cli_arg)
    if not base.is_dir():
        raise FileNotFoundError(
            f"Results bundle not found at: {base}\n"
            f"Download 'NAS_Experiment_Results' (from Google Drive) and either:\n"
            f"  - place it in the project root, or\n"
            f"  - set {ENV_VAR}=/path/to/NAS_Experiment_Results, or\n"
            f"  - pass --results-dir /path/to/NAS_Experiment_Results"
        )
    return base


# ---------------------------------------------------------------------------
# Logical subfolder accessors — map code-level names to the bundle layout.
# ---------------------------------------------------------------------------

def nas_results_dir(base: Path) -> Path:
    """NAS search outputs (best_architecture_*.json, nas_results_*.json)."""
    return base / "nas_results"


def final_training_dir(base: Path) -> Path:
    """Final GAN training runs per strategy (incl. the manual_dcgan baseline)."""
    return base / "final_training_500"


def model_evaluation_dir(base: Path) -> Path:
    """GAN evaluation outputs (comprehensive_metrics, etc.)."""
    return base / "model_evaluation"


def downstream_cnn_dir(base: Path) -> Path:
    """Downstream CNN classifier results (baseline vs. synthetic-augmented)."""
    return base / "downstream_task_CNN"
