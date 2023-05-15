import os
import math
import torch
from PIL import Image

torch.set_printoptions(profile="full")


def get_big_map(path):
    """
    get the big map stored on the UAV locally.
    :param path: stored path.
    :return: big map paths and center coordinates.
    """
    paths = []
    labels = []

    file_path = os.listdir(path)
    file_path.sort()

    for file in file_path:
        full_file_path = os.path.join(path, file)
        paths.append(full_file_path)
        file = file[:-5]
        file = file.split(',')
        labels.append(list(map(eval, [file[0], file[1]])))

    return paths, labels


def get_ideal_route(path):
    """
    get the ideal route.
    :param path: file path that recorded all cluster centers.
    :return: longitude and latitude coordinates of the ideal route.
    """
    route = []
    f = open(path, 'rt')
    for line in f:
        line = line.strip('\n')
        line = line.split(' ')
        route.append(list(map(eval, [line[0], line[1]])))
    route.sort(reverse=True)
    f.close()

    return route


def transform_predicted_label_to_route_idx(path, center):
    """
    transform the predicted label (not sorted) into the position index (sorted) in the ideal route.
    :param path: file path that recorded all cluster centers.
    :param center: predicted label (not sorted),
                    which should be transformed to the position index (sorted) in the ideal route.
    :return: the position index (sorted) in the ideal route.
    """
    # the sorted ideal route
    ideal_route = get_ideal_route(path)
    # not sorted labels
    f = open(path, 'rt')
    i = 0
    for line in f:
        if i == center:
            line = line.strip('\n')
            line = line.split(' ')
            # find the position index (sorted) in the ideal route
            route_idx = ideal_route.index(list(map(eval, [line[0], line[1]])))
            break
        i += 1
    f.close()

    return route_idx


def save_candidates():
    """
    save candidate image for matching-based methods.
    :return:
    """
    if os.path.exists("candidates") is False:
        os.mkdir("candidates")

    paths, labels = get_big_map(path="../bigmap")
    centers = get_ideal_route(path="../processOrder/100/cluster_centre.txt")

    # 50 m away from the ideal position
    lat_diff = 50 * 9e-6
    lon_diff = 50 * 1.043e-5
    number = 0
    for ideal_center in centers:
        candidates = [[ideal_center[0] - lat_diff, ideal_center[1] - lon_diff],
                      [ideal_center[0] - lat_diff, ideal_center[1]],
                      [ideal_center[0] - lat_diff, ideal_center[1] + lon_diff],
                      [ideal_center[0], ideal_center[1] - lon_diff],
                      [ideal_center[0], ideal_center[1]],
                      [ideal_center[0], ideal_center[1] + lon_diff],
                      [ideal_center[0] + lat_diff, ideal_center[1] - lon_diff],
                      [ideal_center[0] + lat_diff, ideal_center[1]],
                      [ideal_center[0] + lat_diff, ideal_center[1] + lon_diff],

                      [ideal_center[0] + lat_diff / 2, ideal_center[1] + lon_diff / 2],
                      [ideal_center[0] + lat_diff / 2, ideal_center[1] - lon_diff / 2],
                      [ideal_center[0] - lat_diff / 2, ideal_center[1] + lon_diff / 2],
                      [ideal_center[0] - lat_diff / 2, ideal_center[1] - lon_diff / 2]
                      ]

        if os.path.exists("candidates/" + str(number)) is False:
            os.mkdir("candidates/" + str(number))

        # screenshot all candidate images from big map
        for candi_i in range(0, len(candidates)):
            if os.path.exists("candidates/" + str(number) + "/" + str(candidates[candi_i][0]) + "," + str(candidates[candi_i][1]) + '.png'):
                continue
            min_dis = math.inf
            idx = -1

            for i in range(0, len(labels)):
                lat_dis = (candidates[candi_i][0] - labels[i][0]) * 111000
                lon_dis = (candidates[candi_i][1] - labels[i][1]) * 111000 * math.cos(labels[i][0] / 180 * math.pi)

                dis = math.sqrt(lat_dis * lat_dis + lon_dis * lon_dis)
                if dis < min_dis:
                    min_dis = dis
                    idx = i

            # find the most match big map and screenshot
            lat_dis = (candidates[candi_i][0] - labels[idx][0]) * 111000
            lon_dis = (candidates[candi_i][1] - labels[idx][1]) * 111000 * math.cos(labels[idx][0] / 180 * math.pi)
            lat_pixel_dis = lat_dis // 0.13986
            lon_pixel_dis = lon_dis // 0.14075
            center = [5005 // 2, 8192 // 2]
            new_lat_pixel = center[0] - lat_pixel_dis
            new_lon_pixel = center[1] + lon_pixel_dis

            pic = Image.open(paths[idx])
            pic = pic.crop((new_lon_pixel - 960 // 2, new_lat_pixel - 540 // 2,
                            new_lon_pixel + 960 // 2, new_lat_pixel + 540 // 2))
            pic = pic.resize((320, 180))
            pic.save("candidates/" + str(number) + "/" + str(candidates[candi_i][0]) + "," + str(candidates[candi_i][1]) + '.png')

        number += 1


if __name__ == '__main__':
    save_candidates()
