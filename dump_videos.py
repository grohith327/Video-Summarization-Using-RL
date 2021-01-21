import cv2
from tqdm.auto import tqdm
import numpy as np
import glob
import os
import gc
import argparse

parser = argparse.ArgumentParser(description="dump videos as npy")
parser.add_argument(
    "--videos_path", default="", type=str, required=True, help="path to videos",
)
parser.add_argument(
    "--save_path", default="", type=str, required=True, help="path to save videos",
)
args = parser.parse_args()

files = glob.glob(os.path.join(args.videos_path, "*.mp4"))
for file in tqdm(files):
    prefix = file.split("/")[-1].split(".")[0]
    save_path = os.path.join(args.save_path, f"{prefix}.npy")
    if os.path.exists(save_path):
        continue
    frames = []
    cap = cv2.VideoCapture(file)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    frames = np.stack(frames)
    np.save(save_path, frames)
    del frames
    gc.collect()
