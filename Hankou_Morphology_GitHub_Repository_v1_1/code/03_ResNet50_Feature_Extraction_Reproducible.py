from __future__ import annotations

from pathlib import Path
import argparse
import csv

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class ComponentDataset(Dataset):
    def __init__(self, root: Path):
        self.records = []
        for building_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            for component_dir in sorted(path for path in building_dir.iterdir() if path.is_dir()):
                for image_path in sorted(component_dir.iterdir()):
                    if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                        self.records.append(
                            (building_dir.name, component_dir.name, image_path.name, image_path)
                        )

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        building, component, filename, path = self.records[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), building, component, filename


def load_model(checkpoint: Path, device: torch.device):
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Identity()

    state_dict = torch.load(checkpoint, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    state_dict = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("fc.")
    }
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            "The locally stored official ImageNet-1K checkpoint was not found. "
            "The model weight file is not included in this public repository."
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = ComponentDataset(args.data_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = load_model(args.checkpoint, device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "building_name", "component_name", "image_name",
        *[f"feature_{index}" for index in range(2048)],
    ]

    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        with torch.inference_mode():
            for images, buildings, components, filenames in loader:
                vectors = model(images.to(device)).cpu().numpy()
                for building, component, filename, vector in zip(
                    buildings, components, filenames, vectors
                ):
                    writer.writerow(
                        [building, component, filename, *vector.tolist()]
                    )


if __name__ == "__main__":
    main()
