"""experiments
Registry of available experiments to run."""
from fake_faces.experiments import (
    baseline,
)

EXPERIMENTS = {
    **{t.slug: t for t in baseline.TRIALS}
}
