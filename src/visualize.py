"""Visualization helpers for five required result figures."""

FIGURE_NAMES = (
    "training_curve",
    "reconstruction_samples",
    "latent_space",
    "metric_comparison",
    "ablation_results",
)


def generate_figures(*args, **kwargs) -> None:
    """Generate the five required figures into the results directory."""
    raise NotImplementedError("Figure generation has not been implemented yet")
