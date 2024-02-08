import torchshow

from dataset_misc import MVTecDataset
from inference import EfficientADInferencer

efficientad = EfficientADInferencer(
    teacher_path="models/teacher_layer1_index1.pth",
    student_path="models/student.pth",
    autoencoder_path="models/autoencoder.pth",
)

dataset = MVTecDataset(dataset_name="mvtec_loco", group="splicing_connectors", phase="test")

result = efficientad.forward(dataset[0])
torchshow.show(result)
