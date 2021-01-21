from tqdm.auto import tqdm
import numpy as np
import glob
import os
from torchvision import models
from torchvision import transforms
import torch
import torch.nn as nn
from PIL import Image
import gc
import argparse
import h5py
import json
from augs import (
    GaussianBlur,
    Cutout,
    CutoutColor,
    CenterCrop,
    Rotate,
    Flip,
    Grayscale,
    Original,
)

parser = argparse.ArgumentParser(description="dump videos as features")
parser.add_argument(
    "--videos_path",
    default="",
    type=str,
    required=True,
    help="path to npy stored videos",
)
parser.add_argument(
    "--save_path", default="", type=str, required=True, help="path to features",
)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32

preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

resnet = models.resnet50(pretrained=True)
modules = list(resnet.children())[:-1]
resnet = nn.Sequential(*modules)
resnet.to(device)
resnet.eval()

files = glob.glob(os.path.join(args.videos_path, "*.npy"))
errors = []
Augmentations = [
    Original(),
    GaussianBlur(),
    Cutout(),
    CutoutColor(),
    CenterCrop(),
    Rotate(),
    Flip(),
    Grayscale(),
]

dataset = h5py.File("datasets/eccv16_dataset_tvsum_google_pool5.h5", "r")
all_picks = dict(
    zip(
        list(dataset.keys()),
        [dataset[key]["picks"][...] for key in list(dataset.keys())],
    )
)
f = open("id_to_key_map_tvsum.json")
id_key_map = json.load(f)
f.close()

for i, file in enumerate(files):
    prefix = file.split("/")[-1].split(".")[0]
    save_path = os.path.join(args.save_path, prefix)
    picks = all_picks[id_key_map[prefix]]
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    try:
        video = np.load(file)
        video = video[picks, :, :, :]
    except:
        errors.append(file)
        continue

    for aug in tqdm(Augmentations, desc=f"Augmenting video {i+1}/{len(files)}"):
        aug_name = aug.__class__.__name__.lower()
        curr_save_path = os.path.join(save_path, f"{prefix}_{aug_name}.pt")

        if os.path.exists(curr_save_path):
            continue

        video_aug = aug(video)

        features = []
        inputs = []

        for image in tqdm(video_aug, desc=aug_name):
            image = Image.fromarray(image.astype(np.uint8))
            image = preprocess(image)
            image = image.unsqueeze(0).to(device)
            inputs.append(image)
            if len(inputs) % batch_size == 0:
                inputs = torch.cat(inputs, 0)
                with torch.no_grad():
                    feat = resnet(inputs)
                features.append(feat.squeeze().cpu())
                inputs = []

        if len(inputs) > 0:
            inputs = torch.cat(inputs, 0)
            with torch.no_grad():
                feat = resnet(inputs)
            features.append(feat.squeeze(-1).squeeze(-1).cpu())

        features = torch.cat(features, 0)
        features = features.view(-1, 2048)
        torch.save(features.cpu(), curr_save_path)
        del features
        gc.collect()

print("Errors")
print(errors)
