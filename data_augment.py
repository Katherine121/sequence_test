import random
import imgaug.augmenters as iaa


def pos_aug(pos):
    # 自身误差0.1m
    random_lat = 9e-7 * random.random()
    random_lon = 1.043e-6 * random.random()

    random_idx = random.randint(a=1, b=2)
    # 随机气流1m
    if random_idx == 1:
        random_lat += 9e-6 * random.random()
        random_lon += 1.043e-5 * random.random()
    # 单向风速5m
    else:
        random_lat += 4.5e-5 * random.random()
        random_lon += 5.217e-5 * random.random()

    return [pos[0] + random_lat, pos[1] + random_lon]


def image_aug(images):
    # 光照
    images = iaa.imgcorruptlike.Brightness(severity=random.randint(a=1, b=5))

    # 遮挡
    random_idx = random.randint(a=1, b=4)
    # 下雨
    if random_idx == 1:
        images = iaa.Rain(speed=(0.1, 0.3))
    # 下雪
    elif random_idx == 2:
        images = iaa.Snowflakes(flake_size=(0.1, 0.4), speed=(0.01, 0.03))
    # 下雾
    elif random_idx == 3:
        images = iaa.Fog()
    else:
        images = iaa.Cutout(nb_iterations=1, size=0.1, fill_per_channel=True)

    return images
