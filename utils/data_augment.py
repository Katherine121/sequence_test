import random
import imgaug.augmenters as iaa
import numpy as np
from PIL import Image
from torch import nn


class ImageAugment(nn.Module):
    def __init__(self, style_idx, shift_idx):
        """
        ImageAugment.
        :param style_idx: the index of style augment.
        :param shift_idx: the index of coordinate shift augment.
        """
        super(ImageAugment, self).__init__()
        self.style_idx = style_idx
        self.shift_idx = shift_idx

        if style_idx == "bright":
            self.style = iaa.imgcorruptlike.Brightness(severity=1)
        elif style_idx == "rain":
            self.style = iaa.Rain(nb_iterations=1, drop_size=0.3, speed=0.3)
        elif style_idx == "snow":
            self.style = iaa.Snowflakes(density=0.01, density_uniformity=0.2,
                                        flake_size=0.1, flake_size_uniformity=0.2,
                                        speed=0.01)
        elif style_idx == "fog":
            self.style = iaa.Fog()
        elif style_idx == "cutout":
            self.style = iaa.Cutout(size=0.2)
        else:
            self.style = None

        if self.shift_idx == "random1":
            self.random_lat = 0.5 * 9e-6
            self.random_lon = 0.5 * 1.043e-5
        elif self.shift_idx == "uni5":
            self.lat_ang = None
            self.lon_ang = None
            self.random_lat = 0.05 * 9e-6
            self.random_lon = 0.05 * 1.043e-5
        else:
            self.random_lat = 0
            self.random_lon = 0

    def forward(self, image):
        """
        forward pass of ImageAugment.
        :param image: the provided input image.
        :return: the image after augmented.
        """
        if self.style is not None:
            image = self.style(image=image)
        return image

    def forward_shift(self):
        """
        forward pass of coordinate shift augment.
        :return: coordinate shift vector.
        """
        if self.shift_idx == "random1":
            random_lat_shift = random.uniform(-self.random_lat, self.random_lat)
            random_lon_shift = random.uniform(-self.random_lon, self.random_lon)
        elif self.shift_idx == "uni5":
            if self.lat_ang is None:
                random_lat_shift = random.uniform(-self.random_lat, self.random_lat)
                random_lon_shift = random.uniform(-self.random_lon, self.random_lon)
                self.lat_ang = 1 if random_lat_shift >= 0 else -1
                self.lon_ang = 1 if random_lon_shift >= 0 else -1
            else:
                if self.lat_ang == 1:
                    random_lat_shift = random.uniform(0, self.random_lat)
                else:
                    random_lat_shift = random.uniform(-self.random_lat, 0)
                if self.lon_ang == 1:
                    random_lon_shift = random.uniform(0, self.random_lon)
                else:
                    random_lon_shift = random.uniform(-self.random_lon, 0)
        else:
            random_lat_shift = 0
            random_lon_shift = 0
        return [random_lat_shift, random_lon_shift]


if __name__ == '__main__':
    image_augment = ImageAugment(style_idx="bright", shift_idx="ori")
    pic = Image.open("../1.png")

    pic = np.array(pic)
    pic = image_augment(pic)
    pic = Image.fromarray(pic)
    pic.save("../2.png")
