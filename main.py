import torchshow

from dataset_misc import MVTecIterableDataset
from inference import EfficientADInferencer

efficientad = EfficientADInferencer()

dataset = MVTecIterableDataset(dataset_name="mvtec_loco", group="splicing_connectors", phase="test")

result = efficientad.forward(dataset[0])
torchshow.show(result)
