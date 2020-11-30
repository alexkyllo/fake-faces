"""experiments
Registry of available experiments to run."""
from fake_faces.experiments import (
    baseline,
    vgg10,
    resnet,
)

EXPERIMENTS = {
    **{t.slug: t for t in baseline.TRIALS},
    **{t.slug: t for t in vgg10.TRIALS},
    **{t.slug: t for t in resnet.TRIALS},
}
