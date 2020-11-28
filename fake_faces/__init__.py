__version__ = '0.1.0'
SHAPE = (128, 128)
BATCH_SIZE = 64
CLASS_MODE = "binary"
CHECKPOINT_FMT = "model.{epoch:02d}-{val_loss:.2f}-{val_accuracy:.2f}.hdf5"
EPOCH_PAT = "model.([0-9]+)"
RESCALE = 1.0 / 255
