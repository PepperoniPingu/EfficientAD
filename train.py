import datasets
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
import models
from PIL import Image

CHANNELS = 100
TRAINING_ITERATIONS = 60_000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def common_preprocess(image: Image) -> torch.Tensor:
    global DEVICE
    res = image.convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((512, 512), antialias=True),s
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomGrayscale(0.1),
    ])
    res = preprocess(res).to(DEVICE)
    return res

def resnet_preprocess(image: torch.Tensor) -> torch.Tensor:
    preprocess = transforms.Compose([
        transforms.Resize((512, 512), antialias=True),
    ])
    return preprocess(image)

def pdn_preprocess(image: torch.Tensor) -> torch.Tensor:
    preprocess = transforms.Compose([
        transforms.Resize((512, 512), antialias=True),
    ])
    return preprocess(image)

@torch.no_grad()
def find_distribution(model: torch.nn.Module, dataset: datasets.IterableDatasetDict, sample_size: int = 10_000) -> tuple:
    # find gaussian distribution of features
    features = None
    for data in dataset["train"].take(sample_size):
        image = common_preprocess(data["image"])
        image = resnet_preprocess(image).unsqueeze(0)
        feature_map = model(image)[-1]
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
    global DEVICE
    global TRAINING_ITERATIONS

    # if the dataset is gated/private, make sure you have run huggingface-cli login
    dataset = datasets.load_dataset("imagenet-1k", trust_remote_code=True, streaming=True)

    target_features_model = timm.create_model("wide_resnet101_2", features_only=True, pretrained=True).cuda()
    # we only care about the first layer
    del target_features_model["layer4"]
    del target_features_model["layer3"]
    del target_features_model["layer2"]
    #del base_descriptions_model["layer1"][2]

    print("finding normal distribution of features...")
    std, mean = find_distribution(target_features_model, dataset, sample_size=100)
    std = std.unsqueeze(1).unsqueeze(1).expand(-1, 128, 128)
    mean = mean.unsqueeze(1).unsqueeze(1).expand(-1, 128, 128)

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    teacher_pdn = models.get_pdn(channels=256).cuda()
    teacher_pdn.train()

    optimizer = torch.optim.Adam(teacher_pdn.parameters(), lr=1e-4, weight_decay=1e-5)

    print("starting training of teacher...")
    image_batch = None
    for data, iteration in zip(dataset["train"], range(TRAINING_ITERATIONS)):

        image = common_preprocess(data["image"]).unsqueeze(0)
        if image_batch is None:
            image_batch = image
        else:
            image_batch = torch.cat((image_batch, image), 0)

        if (iteration + 1) % 16 == 0:
            optimizer.zero_grad()

            target_features = target_features_model(resnet_preprocess(image_batch))[-1].squeeze()
            target_features = torch.sub(target_features, mean) / std

            predicted_features = teacher_pdn(pdn_preprocess(image_batch))[-1]

            loss = torch.mean((target_features - predicted_features)**2) # calculate mean square error
            print(f"iteration: {iteration}  loss: {loss.item()}")

            loss.backward()
            optimizer.step()

        if iteration % 5000 == 0:
            torch.save(teacher_pdn, "models/tmp/teacher.pth")

    torch.save(teacher_pdn, "models/teacher.pth")
    teacher_pdn.eval()
    print("finished training teacher!")

if __name__ == "__main__":
    main()