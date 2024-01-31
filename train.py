import argparse
import random

import datasets
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import models


def common_preprocess(image: Image, device: torch.DeviceObjType) -> torch.Tensor:
    res = image.convert("RGB")
    preprocess = transforms.Compose(
        [
            transforms.Resize((512, 512), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomGrayscale(0.1),
        ]
    )
    res = preprocess(res).to(device)
    return res


def resnet_preprocess(image: torch.Tensor) -> torch.Tensor:
    preprocess = transforms.Compose(
        [
            transforms.Resize((512, 512), antialias=True),
        ]
    )
    return preprocess(image)


def pdn_preprocess(image: torch.Tensor, resize_to: tuple[int, int]) -> torch.Tensor:
    preprocess = transforms.Compose(
        [
            transforms.Resize(resize_to, antialias=True),
        ]
    )
    return preprocess(image)


def distortion_preprocess(image: torch.Tensor) -> torch.Tensor:
    distortion = random.choice([1, 2, 3])
    factor = random.uniform(0.8, 1.2)
    if distortion == 1:
        distorted_image = transforms.functional.adjust_brightness(image, factor)
    elif distortion == 2:
        distorted_image = transforms.functional.adjust_contrast(image, factor)
    elif distortion == 3:
        distorted_image = transforms.functional.adjust_saturation(image, factor)

    return distorted_image


@torch.no_grad()
def find_distribution(
    model: torch.nn.Module,
    dataset: datasets.IterableDatasetDict,
    device: torch.DeviceObjType,
    sample_size: int = 10_000,
) -> tuple:
    # find gaussian distribution of features
    features = None
    for data in dataset["train"].take(sample_size):
        image = common_preprocess(data["image"], device)
        image = resnet_preprocess(image).unsqueeze(0)
        feature_map = model.forward(image)
        feature_map = torch.flatten(feature_map, start_dim=2)
        if features is None:
            features = feature_map
        else:
            features = torch.cat((features, feature_map), 0)
    features = torch.movedim(features, 0, 1)
    features = torch.flatten(features, start_dim=1)
    std, mean = torch.std_mean(features, dim=1)
    return std, mean


def train_teacher(
    channels: int,
    dataset: datasets.IterableDatasetDict,
    device: torch.DeviceObjType,
    epochs: int = 10_000,
    batch_size: int = 8,
    wideresnet_feature_layer: str = "layer1",
    wideresnet_feature_layer_index: int = 1,
    tensorboard_writer: SummaryWriter | None = None,
) -> torch.nn.Module:
    target_features_model = models.ChoppedWideResNet(
        channels=channels, layer_to_extract_from=wideresnet_feature_layer, layer_index=wideresnet_feature_layer_index
    ).to(device)
    target_features_model.eval()
    target_features_output_shape = target_features_model.forward(
        resnet_preprocess(torch.rand([1, 3, 1, 1]).to(device))
    ).shape

    print("finding normal distribution of features...")
    std, mean = find_distribution(target_features_model, dataset, device, sample_size=50)
    std = std.unsqueeze(1).unsqueeze(1).expand(target_features_output_shape)
    mean = mean.unsqueeze(1).unsqueeze(1).expand(target_features_output_shape)

    teacher_pdn = models.PatchDescriptionNetwork(channels=channels).to(device)
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
                target_features = torch.sub(target_features, mean)
                target_features = target_features / std
                target_features = torch.nan_to_num(target_features)

            predicted_features = teacher_pdn.forward(
                pdn_preprocess(image_batch, (target_features_output_shape[2] * 4, target_features_output_shape[3] * 4))
            )

            loss = torch.mean((target_features - predicted_features) ** 2)  # calculate mean square error
            print(f"epoch: {int(iteration/batch_size)}/{epochs}  loss: {loss.item()}")
            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar("teacher training", loss.item(), int(iteration / batch_size))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            image_batch = None

        if iteration % 5000 == 0:
            torch.save(teacher_pdn, "models/tmp/teacher.pth")

    torch.save(teacher_pdn, "models/teacher.pth")
    print("finished training teacher!")


def train_autoencoder(
    channels: int,
    dataset: datasets.IterableDatasetDict,
    device: torch.DeviceObjType,
    teacher: torch.nn.Module,
    epochs: int = 10_000,
    batch_size: int = 8,
    tensorboard_writer: SummaryWriter | None = None,
) -> torch.nn.Module:
    autoencoder = models.AutoEncoder(channels=channels).to(device)
    autoencoder.train()
    teacher.eval()

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-4, weight_decay=1e-5)

    for data, iteration in zip(dataset["train"], range(batch_size * epochs)):
        image = common_preprocess(data["image"], device).unsqueeze(0)
        image = distortion_preprocess(image)
        if image_batch is None:
            image_batch = image
        else:
            image_batch = torch.cat((image_batch, image), 0)

        if (iteration + 1) % batch_size == 0:
            with torch.no_grad():
                teacher_result = teacher.forward(image_batch)
            autoencoder_result = autoencoder.forward(image_batch)

            loss = torch.mean((teacher_result - autoencoder_result) ** 2)
            print(f"epoch: {int(iteration/batch_size)}/{epochs}  loss: {loss.item()}")
            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar("autoencoder training", loss.item(), int(iteration / batch_size))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            image_batch = None


def train_student(
    channels: int,
    dataset: datasets.IterableDatasetDict,
    device: torch.DeviceObjType,
    teacher: torch.nn.Module,
    epochs: int = 10_000,
    batch_size: int = 8,
    tensorboard_writer: SummaryWriter | None = None,
) -> torch.nn.Module:
    student_pdn = models.PatchDescriptionNetwork(channels=channels * 2).to(device)
    student_pdn.train()
    teacher.eval()

    optimizer = torch.optim.Adam(student_pdn.parameters(), lr=1e-4, weight_decay=1e-5)

    for data, iteration in zip(dataset["train"], range(batch_size * epochs)):
        image = common_preprocess(data["image"], device).unsqueeze(0)
        if image_batch is None:
            image_batch = image
        else:
            image_batch = torch.cat((image_batch, image), 0)

        if (iteration + 1) % batch_size == 0:
            teacher_result = teacher.forward(image_batch)
            student_result = student_pdn.forward(image_batch)

            loss = torch.mean((teacher_result - student_result) ** 2)
            print(f"epoch: {int(iteration/batch_size)}/{epochs}  loss: {loss.item()}")
            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar("student training", loss.item(), int(iteration / batch_size))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            image_batch = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", choices=["cpu", "cuda"])
    parser.add_argument("--skip_teacher", action="store_true", default=False)
    parser.add_argument("--skip_autoencoder", action="store_true", default=False)
    parser.add_argument("--skip_student", action="store_true", default=False)
    args = parser.parse_args()
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    channels = 256

    tensorboard_writer = SummaryWriter()

    # if the dataset is gated/private, make sure you have run huggingface-cli login
    generic_dataset = datasets.load_dataset("imagenet-1k", trust_remote_code=True, streaming=True)

    if not args.skip_teacher:
        teacher_pdn = train_teacher(
            channels=channels,
            dataset=generic_dataset,
            device=device,
            tensorboard_writer=tensorboard_writer,
            wideresnet_feature_layer_index=0,
        )

    del generic_dataset
    good_dataset = datasets.load_dataset(
        "imagenet-1k", trust_remote_code=True, streaming=True
    )  # change this to another appropriate dataset

    if not args.skip_autoencoder:
        autoencoder = train_autoencoder(
            channels=channels,
            dataset=good_dataset,
            device=device,
            teacher=teacher_pdn,
            tensorboard_writer=tensorboard_writer,
        )

    if not args.skip_student:
        student_pdn = train_student(
            channels=channels,
            dataset=good_dataset,
            device=device,
            teacher=teacher_pdn,
            tensorboard_writer=tensorboard_writer,
        )

    tensorboard_writer.close()


if __name__ == "__main__":
    main()
