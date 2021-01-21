import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from PolicyGradient import PolicyNet
from dataset import LstmDataset
from tools import generate_summary

import os
import glob


class args:
    cnn_feat = "resnet50"
    train_cnn = False
    device = torch.device("cuda")


def save_video(frames, filename):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(filename, fourcc, 29, (width, height))
    for i in range(len(frames)):
        writer.write(frames[i])
    writer.release()


@torch.no_grad()
def save_pred(args, model, data):
    features = data["feature"].to(args.device).unsqueeze(0)
    change_points = data["change_points"]
    nfps = data["nfps"].tolist()
    picks = data["picks"].numpy()
    n_frames = data["n_frames"]
    id = data["id"]

    probs, _ = model(features)
    probs = probs.squeeze().cpu().numpy()

    summary = generate_summary(probs, change_points, n_frames, nfps, picks)
    video = np.load(f"videos_npy/{id}.npy")
    summary_video = video[summary.astype(np.bool)]
    save_video(summary_video, f"tvsum_summaries/{id}.avi")


model = PolicyNet(args).to(args.device)
ckpt = torch.load("models/resnet50_augs_PPO_seed_4/model_3.pth")
model.load_state_dict(ckpt["model"])

idxs = [29, 12, 47, 43, 30, 8, 19, 25, 27, 11]
paths = glob.glob("cnn_feats/*")
infer_dataset = LstmDataset(paths)

for i in tqdm(idxs):
    data = infer_dataset[i]
    save_pred(args, model, data)
