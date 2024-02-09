import argparse
import random

import datasets
import torch
import yaml
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import models
from dataset_misc import (
    ConvertedHuggingFaceIterableDataset,
    MVTecIterableDataset,
    TensorConvertedIterableDataset,
    TransformedIterableDataset,
)


def resnet_preprocess(image: torch.Tensor) -> torch.Tensor:
    preprocess = transforms.Compose(
        [
            transforms.Resize((512, 512), antialias=True),
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
    dataset: IterableDataset,
    device: torch.DeviceObjType,
    sample_size: int = 100,
) -> tuple:
    # find gaussian distribution of features
    preprocess = transforms.Compose(
        [
            transforms.Resize((512, 512), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataloader = DataLoader(TransformedIterableDataset(dataset, preprocess), batch_size=sample_size)

    batch = next(iter(dataloader)).to(device)
    features = model.forward(batch)
    features = torch.movedim(features, 0, 1)
    features = torch.flatten(features, start_dim=1)
    std, mean = torch.std_mean(features, dim=1, unbiased=False)

    return std, mean


def train_teacher(
    channels: int,
    generic_dataset: IterableDataset,
    good_dataset: IterableDataset,
    device: torch.DeviceObjType,
    batches: int = 10_000,
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
    std, mean = find_distribution(target_features_model, generic_dataset, device, sample_size=50)
    std = std.unsqueeze(1).unsqueeze(1).expand(target_features_output_shape)
    mean = mean.unsqueeze(1).unsqueeze(1).expand(target_features_output_shape)

    common_preprocess = transforms.Compose(
        [
            transforms.Resize((512, 512), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomGrayscale(0.1),
        ]
    )

    pdn_preprocess = transforms.Compose(
        [
            transforms.Resize(
                (target_features_output_shape[2] * 4, target_features_output_shape[3] * 4), antialias=True
            ),
        ]
    )

    dataloader = DataLoader(TransformedIterableDataset(generic_dataset, common_preprocess), batch_size=batch_size)

    teacher_pdn = models.PatchDescriptionNetwork(channels=channels).to(device)
    teacher_pdn.train()

    optimizer = torch.optim.Adam(teacher_pdn.parameters(), lr=1e-4, weight_decay=1e-5)

    print("starting training of teacher...")
    for image_batch, batch in zip(dataloader, range(batches)):
        image_batch = image_batch.to(device)

        with torch.no_grad():
            target_features = target_features_model.forward(resnet_preprocess(image_batch))
            target_features = torch.sub(target_features, mean)
            target_features = target_features / std
            target_features = torch.nan_to_num(target_features)

        predicted_features = teacher_pdn.forward(pdn_preprocess(image_batch))

        loss = torch.mean((target_features - predicted_features) ** 2)  # calculate mean square error
        print(f"batch: {batch}/{batches}  loss: {loss.item()}")
        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar("teacher training", loss.item(), batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1000 == 0:
            torch.save(teacher_pdn, "models/tmp/teacher.pth")

    torch.save(teacher_pdn, "models/generic_teacher.pth")
    print("finished training teacher!")

    print("normalizing teacher...")
    preprocess = transforms.Compose(
        [
            transforms.Resize((512, 512), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataloader = DataLoader(TransformedIterableDataset(good_dataset, preprocess), batch_size=batch_size)
    teacher_pdn = models.NormalizedPatchDescriptionNetwork(teacher_pdn).train().to(device)
    for image_batch, batch in zip(dataloader, len(dataloader)):
        teacher_pdn.forward(image_batch)
        print(f"batch: {batch}/{len(dataloader)}")

    torch.save(teacher_pdn, "models/teacher.pth")
    print("finished normalizing teacher!")

    return teacher_pdn


def train_autoencoder(
    channels: int,
    dataset: IterableDataset,
    device: torch.DeviceObjType,
    teacher: torch.nn.Module,
    epochs: int = 5_000,
    batch_size: int = 8,
    tensorboard_writer: SummaryWriter | None = None,
) -> torch.nn.Module:
    autoencoder = models.AutoEncoder(channels=channels).to(device)
    autoencoder.train()
    teacher.eval()

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-4, weight_decay=1e-5)

    preprocess = transforms.Compose(
        [
            transforms.Resize((256, 256), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomGrayscale(0.1),
        ]
    )

    dataloader = DataLoader(TransformedIterableDataset(dataset, preprocess), batch_size=batch_size)

    print("starting training of autoencoder...")
    for epoch in range(epochs):
        for image_batch, batch in zip(dataloader, range(len(dataloader))):
            image_batch = image_batch.to(device)
            image_batch = distortion_preprocess(image_batch)

            with torch.no_grad():
                teacher_result = teacher.forward(image_batch)
            autoencoder_result = autoencoder.forward(image_batch)

            loss = torch.mean((teacher_result - autoencoder_result) ** 2)

            total_batch = batch + epoch * len(dataloader)
            print(f"batch: {total_batch}/{epochs * len(dataloader)}  loss: {loss.item()}")
            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar("autoencoder training", loss.item(), total_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(autoencoder, "models/tmp/autoencoder.pth")

    torch.save(autoencoder, "models/autoencoder.pth")
    print("finished training autoencoder!")

    return autoencoder


def train_student(
    channels_teacher: int,
    channels_autoencoder: int,
    good_dataset: IterableDataset,
    generic_dataset: IterableDataset,
    device: torch.DeviceObjType,
    teacher: torch.nn.Module,
    autoencoder: torch.nn.Module,
    epochs: int = 1_300,
    batch_size: int = 8,
    tensorboard_writer: SummaryWriter | None = None,
) -> torch.nn.Module:
    student_pdn = models.PatchDescriptionNetwork(channels=channels_teacher + channels_autoencoder).to(device)
    student_pdn.train()
    teacher.eval()
    autoencoder.eval()

    optimizer = torch.optim.Adam(student_pdn.parameters(), lr=1e-4, weight_decay=1e-5)

    preprocess = transforms.Compose(
        [
            transforms.Resize(
                (256, 256), antialias=True
            ),  # quantile will not work with 512x512, TODO: sample random values for quantile calculation
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomGrayscale(0.1),
        ]
    )

    dataloader_good = DataLoader(TransformedIterableDataset(good_dataset, preprocess), batch_size=batch_size)
    dataloader_generic = DataLoader(TransformedIterableDataset(generic_dataset, preprocess))

    print("starting training of student...")
    for epoch in range(epochs):
        for image_batch, generic_image, batch in zip(dataloader_good, dataloader_generic, range(len(dataloader_good))):
            image_batch = image_batch.to(device)
            image_batch = distortion_preprocess(image_batch)
            generic_image = generic_image.to(device)

            with torch.no_grad():
                teacher_result = teacher.forward(image_batch)
                autoencoder_result = autoencoder.forward(image_batch)
            student_result = student_pdn.forward(image_batch)

            pdn_student_distance = (teacher_result - student_result[:, :channels_teacher, :, :]) ** 2
            pdn_student_quantile = torch.quantile(pdn_student_distance, q=0.999)
            pdn_student_hard_loss = torch.mean(pdn_student_distance[pdn_student_distance >= pdn_student_quantile])

            pdn_student_penalty_result = student_pdn.forward(generic_image)[:, :channels_teacher, :, :]
            pdn_student_penalty_loss = torch.mean(pdn_student_penalty_result**2)

            pdn_student_loss = pdn_student_hard_loss + pdn_student_penalty_loss

            autoencoder_student_loss = torch.mean(
                (autoencoder_result - student_result[:, channels_teacher:, :, :]) ** 2
            )
            total_loss = pdn_student_loss + autoencoder_student_loss

            total_batch = batch + epoch * len(dataloader_good)
            print(f"batch: {total_batch}/{epochs * len(dataloader_good)}  loss: {total_loss.item()}")
            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar("student training", total_loss.item(), total_batch)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        torch.save(student_pdn, "models/tmp/student.pth")

    torch.save(student_pdn, "models/student.pth")
    print("finished training student!")

    return student_pdn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", choices=["cpu", "cuda"])
    parser.add_argument("--skip-teacher", action="store_true", default=False)
    parser.add_argument("--skip-autoencoder", action="store_true", default=False)
    parser.add_argument("--skip-student", action="store_true", default=False)
    parser.add_argument("--model-config", action="store", default="model_config.yaml")
    args = parser.parse_args()
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    tensorboard_writer = SummaryWriter()

    torch.manual_seed(1337)

    with open(args.model_config) as config_file:
        model_config = yaml.safe_load(config_file)

    # if the dataset is gated/private, make sure you have run huggingface-cli login
    generic_dataset = TensorConvertedIterableDataset(
        ConvertedHuggingFaceIterableDataset(
            datasets.load_dataset("imagenet-1k", trust_remote_code=True, streaming=True)["train"]
        )
    )
    good_dataset = TensorConvertedIterableDataset(
        MVTecIterableDataset(dataset_name="mvtec_loco", group="splicing_connectors", phase="train")
    )

    if args.skip_teacher:
        teacher_pdn = torch.load(model_config["teacher_path"], map_location=device)
    else:
        teacher_pdn = train_teacher(
            channels=model_config["out_channels"]["teacher"],
            generic_dataset=generic_dataset,
            good_dataset=good_dataset,
            device=device,
            tensorboard_writer=tensorboard_writer,
            wideresnet_feature_layer_index=0,
        )

    if args.skip_autoencoder:
        autoencoder = torch.load(model_config["autoencoder_path"], map_location=device)
    else:
        autoencoder = train_autoencoder(
            channels=model_config["out_channels"]["autoencoder"],
            dataset=good_dataset,
            device=device,
            teacher=teacher_pdn,
            tensorboard_writer=tensorboard_writer,
        )

    if not args.skip_student:
        student_pdn = train_student(
            channels_teacher=model_config["out_channels"]["teacher"],
            channels_autoencoder=model_config["out_channels"]["autoencoder"],
            good_dataset=good_dataset,
            generic_dataset=generic_dataset,
            device=device,
            teacher=teacher_pdn,
            autoencoder=autoencoder,
            tensorboard_writer=tensorboard_writer,
        )

    tensorboard_writer.close()


if __name__ == "__main__":
    main()
