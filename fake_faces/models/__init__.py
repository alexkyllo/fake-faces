from fake_faces.models.vgg10 import VGG10
from fake_faces.models.baseline import Baseline
from fake_faces.models.baseline_batchnorm import BaselineBatchNorm

MODELS = dict(
    baseline = Baseline,
    vgg10 = VGG10,
    batchnorm = BaselineBatchNorm,
)
