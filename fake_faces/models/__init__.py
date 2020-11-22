from fake_faces.models.vgg10 import VGG10
from fake_faces.models.vgg16 import VGG16
from fake_faces.models.densenet121 import DenseNet121
from fake_faces.models.baseline import Baseline
from fake_faces.models.baseline_batchnorm import BaselineBatchNorm

MODELS = dict(
    baseline = Baseline,
    vgg10 = VGG10,
    vgg16 = VGG16,
    densenet121 = DenseNet121,
    batchnorm = BaselineBatchNorm,
)
