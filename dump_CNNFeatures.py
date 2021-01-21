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
for i, file in enumerate(files):
    prefix = file.split("/")[-1].split(".")[0]
    save_path = os.path.join(args.save_path, f"{prefix}.pt")
    if os.path.exists(save_path):
        continue

    try:
        images = np.load(file)
    except:
        errors.append(file)
        continue
    features = []
    inputs = []

    for image in tqdm(images, desc=f"Video {i+1}/{len(files)}"):
        image = Image.fromarray(image)
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
    torch.save(features.cpu(), save_path)
    del features
    gc.collect()

print("Errors")
print(errors)
