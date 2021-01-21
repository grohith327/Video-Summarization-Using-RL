import os
import torch
import h5py
import json
import glob
from torch.utils import data
from utils import get_filename
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class LstmDataset(data.Dataset):
    def __init__(self, paths, cnn_feat="resnet50", dataset="tvsum", use_augs=False):
        super().__init__()
        self.paths = paths
        self.cnn_feat = cnn_feat
        self.use_augs = use_augs

        if dataset == "tvsum":
            self.dataset = h5py.File(
                "datasets/eccv16_dataset_tvsum_google_pool5.h5", "r"
            )
        else:
            self.dataset = h5py.File(
                "datasets/eccv16_dataset_summe_google_pool5.h5", "r"
            )

        f = open(f"id_to_key_map_{dataset}.json")
        self.id_key_map = json.load(f)
        f.close()

        self.change_points = dict(
            zip(
                list(self.dataset.keys()),
                [
                    self.dataset[key]["change_points"][...]
                    for key in list(self.dataset.keys())
                ],
            )
        )

        self.nfps = dict(
            zip(
                list(self.dataset.keys()),
                [
                    self.dataset[key]["n_frame_per_seg"][...]
                    for key in list(self.dataset.keys())
                ],
            )
        )

        self.picks = dict(
            zip(
                list(self.dataset.keys()),
                [self.dataset[key]["picks"][...] for key in list(self.dataset.keys())],
            )
        )

        self.features = dict(
            zip(
                list(self.dataset.keys()),
                [
                    self.dataset[key]["features"][...]
                    for key in list(self.dataset.keys())
                ],
            )
        )

        self.user_summary = dict(
            zip(
                list(self.dataset.keys()),
                [
                    self.dataset[key]["user_summary"][...]
                    for key in list(self.dataset.keys())
                ],
            )
        )

        self.n_frames = dict(
            zip(
                list(self.dataset.keys()),
                [
                    self.dataset[key]["n_frames"][()]
                    for key in list(self.dataset.keys())
                ],
            )
        )

    def __getitem__(self, i):
        path = self.paths[i]
        id = get_filename(path)

        key = self.id_key_map[id]
        user_summary = self.user_summary[key]
        change_points = self.change_points[key]
        nfps = self.nfps[key]
        picks = torch.LongTensor(self.picks[key])
        n_frames = self.n_frames[key]

        if self.cnn_feat == "resnet50":
            if self.use_augs:
                files = f"aug_cnn_feats/{id}/*"
                files = sorted(glob.glob(files))
                index = np.random.randint(8)
                path = files[index]
                feature = torch.load(path)
            else:
                feature = torch.load(path)
                feature = feature[picks, :]
        else:
            feature = self.features[key]

        return {
            "feature": feature,
            "user_summary": user_summary,
            "id": id,
            "change_points": change_points,
            "nfps": nfps,
            "picks": picks,
            "n_frames": n_frames,
        }

    def __len__(self):
        return len(self.paths)


class VideoDataset(data.Dataset):
    def __init__(self, paths, dataset="tvsum"):
        self.paths = list(
            map(
                lambda x: "videos_npy/" + x.split("/")[-1].split(".")[0] + ".npy", paths
            )
        )

        f = open(f"id_to_key_map_{dataset}.json")
        self.id_key_map = json.load(f)
        f.close()

        self.d_name = dataset
        if dataset == "tvsum":
            self.dataset = h5py.File(
                "datasets/eccv16_dataset_tvsum_google_pool5.h5", "r"
            )
        else:
            self.dataset = h5py.File(
                "datasets/eccv16_dataset_summe_google_pool5.h5", "r"
            )

        self.change_points = dict(
            zip(
                list(self.dataset.keys()),
                [
                    self.dataset[key]["change_points"][...]
                    for key in list(self.dataset.keys())
                ],
            )
        )

        self.nfps = dict(
            zip(
                list(self.dataset.keys()),
                [
                    self.dataset[key]["n_frame_per_seg"][...]
                    for key in list(self.dataset.keys())
                ],
            )
        )

        self.picks = dict(
            zip(
                list(self.dataset.keys()),
                [self.dataset[key]["picks"][...] for key in list(self.dataset.keys())],
            )
        )

        self.user_summary = dict(
            zip(
                list(self.dataset.keys()),
                [
                    self.dataset[key]["user_summary"][...]
                    for key in list(self.dataset.keys())
                ],
            )
        )

        self.n_frames = dict(
            zip(
                list(self.dataset.keys()),
                [
                    self.dataset[key]["n_frames"][()]
                    for key in list(self.dataset.keys())
                ],
            )
        )

        self.transforms = self.get_transforms()

    def get_transforms(self):
        return A.Compose(
            [
                A.Resize(224, 224, p=1.0),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0
                ),
                ToTensorV2(),
            ]
        )

    def __getitem__(self, i):
        path = self.paths[i]
        id = get_filename(path)

        key = self.id_key_map[id]
        user_summary = self.user_summary[key]
        change_points = self.change_points[key]
        nfps = self.nfps[key]
        picks = self.picks[key]
        n_frames = self.n_frames[key]

        video = np.load(path)
        video = video[picks, :, :, :]

        if self.transforms:
            transformed_images = []
            for image in video:
                image = self.transforms(image=image)["image"]
                transformed_images.append(image)
            video = torch.stack(transformed_images)

        return {
            "feature": video,
            "user_summary": user_summary,
            "id": id,
            "change_points": change_points,
            "nfps": nfps,
            "picks": picks,
            "n_frames": n_frames,
        }

    def __len__(self):
        return len(self.paths)
