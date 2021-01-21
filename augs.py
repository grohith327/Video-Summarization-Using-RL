import cv2
import numpy as np
import albumentations as A
import random
import copy


class Original:
    def __call__(self, video):
        return video


class GaussianBlur:
    def __init__(self, kernel_size=5):
        self.kernel_size = kernel_size

    def __call__(self, video):
        video = copy.deepcopy(video)
        transformed_video = []
        for img in video:
            img = cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), 0)
            transformed_video.append(img)
        transformed_video = np.stack(transformed_video)
        return transformed_video


class Cutout:

    """
    Adapted from https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
    """

    def __init__(self, n_holes=4, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, video):
        video = copy.deepcopy(video)
        h = video.shape[1]
        w = video.shape[2]
        hole_points = []
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            hole_points.append([x, y])

        transformed_video = []
        for img in video:

            mask = np.ones((h, w), np.float32)

            for n in range(self.n_holes):
                x, y = hole_points[n]

                y1 = np.clip(y - self.length // 2, 0, h)
                y2 = np.clip(y + self.length // 2, 0, h)
                x1 = np.clip(x - self.length // 2, 0, w)
                x2 = np.clip(x + self.length // 2, 0, w)

                mask[y1:y2, x1:x2] = 0.0

            mask = np.expand_dims(mask, axis=-1).repeat(3, axis=-1)
            img = img * mask
            transformed_video.append(img)

        transformed_video = np.stack(transformed_video)
        return transformed_video


class CutoutColor:
    def __init__(self, n_holes=4, length=16, color=None):
        self.n_holes = n_holes
        self.length = length
        self.color = self.random_color() if color is None else color

    def random_color(self):
        rgbl = [255, 0, 0]
        random.shuffle(rgbl)
        return tuple(rgbl)

    def __call__(self, video):
        video = copy.deepcopy(video)
        h = video.shape[1]
        w = video.shape[2]
        hole_points = []
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            hole_points.append([x, y])

        transformed_video = []
        for img in video:

            for n in range(self.n_holes):
                x, y = hole_points[n]

                y1 = np.clip(y - self.length // 2, 0, h)
                y2 = np.clip(y + self.length // 2, 0, h)
                x1 = np.clip(x - self.length // 2, 0, w)
                x2 = np.clip(x + self.length // 2, 0, w)

                img[y1:y2, x1:x2, :] = self.color

            transformed_video.append(img)

        transformed_video = np.stack(transformed_video)
        return transformed_video


class Rotate:
    def __call__(self, video):
        video = copy.deepcopy(video)
        transformed_video = []
        for img in video:
            img = np.rot90(img)
            transformed_video.append(img)
        transformed_video = np.stack(transformed_video)
        return transformed_video


class Flip:
    def __call__(self, video):
        video = copy.deepcopy(video)
        transformed_video = []
        for img in video:
            img = np.fliplr(img)
            transformed_video.append(img)
        transformed_video = np.stack(transformed_video)
        return transformed_video


class CenterCrop:
    def __init__(self, height=180, width=320):
        self.transform = A.Compose([A.CenterCrop(height=height, width=width, p=1.0)])

    def __call__(self, video):
        video = copy.deepcopy(video)
        transformed_video = []
        for img in video:
            img = self.transform(image=img)["image"]
            transformed_video.append(img)
        transformed_video = np.stack(transformed_video)
        return transformed_video


class Grayscale:
    def __init__(self, height=180, width=320):
        self.transform = A.Compose([A.ToGray(p=1.0)])

    def __call__(self, video):
        video = copy.deepcopy(video)
        transformed_video = []
        for img in video:
            img = self.transform(image=img)["image"]
            transformed_video.append(img)
        transformed_video = np.stack(transformed_video)
        return transformed_video
