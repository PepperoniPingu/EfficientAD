import datasets
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
import models
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

def common_preprocess(image: Image, device: torch.DeviceObjType) -> torch.Tensor:
    res = image.convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((512, 512), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomGrayscale(0.1),
    ])
    res = preprocess(res).to(device)
    return res

def resnet_preprocess(image: torch.Tensor) -> torch.Tensor:
    preprocess = transforms.Compose([
        transforms.Resize((512, 512), antialias=True),
    ])
    return preprocess(image)

def pdn_preprocess(image: torch.Tensor) -> torch.Tensor:
    preprocess = transforms.Compose([
        transforms.Resize((256, 256), antialias=True),
    ])
    return preprocess(image)

@torch.no_grad()
def find_distribution(model: torch.nn.Module, feature_layer: int, dataset: datasets.IterableDatasetDict, device: torch.DeviceObjType, sample_size: int = 10_000) -> tuple:
    # find gaussian distribution of features
    features = None
    for data in dataset["train"].take(sample_size):
        image = common_preprocess(data["image"], device)
        image = resnet_preprocess(image).unsqueeze(0)
        feature_map = model.forward(image)[feature_layer]
        feature_map = torch.flatten(feature_map, start_dim=2)
        if features is None:
            features = feature_map
        else:
            features = torch.cat((features, feature_map), 0)
    features = torch.movedim(features, 0, 1)
    features = torch.flatten(features, start_dim=1)
    std, mean = torch.std_mean(features, dim=1)
    return std, mean

def main():
    channels = 100
    epochs = 10_000
    batch_size = 8
    widresnet_feature_layer = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    writer = SummaryWriter()

    # if the dataset is gated/private, make sure you have run huggingface-cli login
    dataset = datasets.load_dataset("imagenet-1k", trust_remote_code=True, streaming=True)

    target_features_model = timm.create_model("wide_resnet101_2", features_only=True, pretrained=True).to(device)

    print("finding normal distribution of features...")
    std, mean = find_distribution(target_features_model, widresnet_feature_layer, dataset, device sample_size=100)
    std = std.unsqueeze(1).unsqueeze(1).expand(-1, 64, 64)[:256,:,:]
    mean = mean.unsqueeze(1).unsqueeze(1).expand(-1, 64, 64)[:256,:,:]

    teacher_pdn = models.get_pdn(channels=256).to(device)
    teacher_pdn.train()

    optimizer = torch.optim.Adam(teacher_pdn.parameters(), lr=1e-4, weight_decay=1e-5)

    print("starting training of teacher...")
    image_batch = None
    for data, iteration in zip(dataset["train"], range(batch_size * epochs)):

        image = common_preprocess(data["image"], device).unsqueeze(0)
        if image_batch is None:
            image_batch = image
        else:
            image_batch = torch.cat((image_batch, image), 0)

        if (iteration + 1) % batch_size == 0:
            with torch.no_grad():
                target_features = target_features_model.forward(resnet_preprocess(image_batch))
                target_features = target_features[widresnet_feature_layer] # select features from layer 1
                target_features = target_features[:,:256,:,:]
                target_features = torch.sub(target_features, mean) / std

            predicted_features = teacher_pdn.forward(pdn_preprocess(image_batch))

            loss = torch.mean((target_features - predicted_features)**2) # calculate mean square error
            print(f"epoch: {int(iteration/batch_size)}/{epochs}  loss: {loss.item()}")
            writer.add_scalar('loss', loss.item(), int(iteration/batch_size))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            image_batch = None

        if iteration % 5000 == 0:
            torch.save(teacher_pdn, "models/tmp/teacher.pth")

    torch.save(teacher_pdn, "models/teacher.pth")
    teacher_pdn.eval()
    print("finished training teacher!")

    writer.close()

if __name__ == "__main__":
    main()