import argparse
import math
import os
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as torchvision_models
from PIL import Image
from selenium import webdriver
import time
from facaformer import FACAFormer

torch.set_printoptions(precision=8)

parser = argparse.ArgumentParser(description='PyTorch Testing')

parser.add_argument('--num_classes1', default=100, type=int,
                    metavar='N', help='the number of milestone labels')
parser.add_argument('--num_classes2', default=2, type=int,
                    metavar='N', help='the number of angle labels (latitude and longitude)')
parser.add_argument('--len', default=6, type=int,
                    metavar='LEN', help='the number of model input sequence length')

parser.add_argument('--dataset-path', default='../../sequence/processOrder/order', type=str,
                    metavar='PATH', help='path to dataset')
parser.add_argument('--path', default="74769/30.3094675,119.923354.png",
                    type=str,
                    metavar='PATH',
                    help='the frame path of initial position')
parser.add_argument('--dest_path', default="74769/30.2619437,119.947853.png",
                    type=str,
                    metavar='PATH',
                    help='the frame path of destination')
parser.add_argument('--last_pos', default=[30.3094675, 119.923354],
                    type=list,
                    metavar='ANGLE',
                    help='last position: lat and lon')
parser.add_argument('--dest_pos', default=[30.2619437, 119.947853],
                    type=list,
                    metavar='ANGLE',
                    help='destination position: lat and lon')


def main():
    """
    real-time testing process control: loading model, dataset, screenshot.
    :return:
    """
    args = parser.parse_args()

    # reduce CPU usage, use it after the model is loaded onto the GPU
    torch.set_num_threads(1)
    cudnn.benchmark = True

    # compute steering angle
    lat_diff = (args.dest_pos[0] - args.last_pos[0]) * 111000
    lon_diff = (args.dest_pos[1] - args.last_pos[1]) * 111000 * math.cos(args.last_pos[0] / 180 * math.pi)
    angle = [lat_diff / math.sqrt(lat_diff * lat_diff + lon_diff * lon_diff),
             lon_diff / math.sqrt(lat_diff * lat_diff + lon_diff * lon_diff)]

    ang = math.atan(angle[0] / angle[1]) * 180 / math.pi
    if lat_diff >= 0 and lon_diff <= 0:
        ang += 180
    elif lat_diff <= 0 and lon_diff <= 0:
        ang -= 180

    # real-time testing
    real_time_test(angle, ang, args)


def real_time_test(angle, ang, args):
    """
    real-time testing process.
    :param val_transform: torchvision.transforms.
    :param model: saved checkpoint.
    :param args:
    :return: actually arrived destination.
    """

    # calculate the moving distance
    dis = 30
    lat_delta = dis * angle[0]
    lon_delta = dis * angle[1]
    lat_delta = float(lat_delta / 111000)
    lon_delta = float(lon_delta / 111000 / math.cos(args.last_pos[0] / 180 * math.pi))

    with torch.no_grad():
        for i in range(0, 156):
            print(i)
            # calculate the new position
            new_lat = args.last_pos[0] + lat_delta
            new_lon = args.last_pos[1] + lon_delta
            print("new_lat: " + str(new_lat))
            print("new_lon: " + str(new_lon))

            # screenshot new frame
            coords = str(new_lat) + "," + str(new_lon)
            path = args.path[0: 6] + coords + '.png'
            if os.path.exists(path) is False:
                path = screenshot(coords, path, ang)

            # append new frame and new angle
            args.path = path
            args.last_pos = [new_lat, new_lon]

    return new_lat, new_lon


def screenshot(coords, path, ang):
    """
    screenshot from Google Earth API.
    :param coords: position needed to be screenshot.
    :param path: saved path of new frame.
    :param ang: rotated angle of new frame.
    :return: saved path of new frame.
    """
    DRIVER = 'chromedriver'
    option = webdriver.ChromeOptions()
    option.add_argument('headless')
    option.add_argument('start-maximized')
    driver = webdriver.Chrome(executable_path=DRIVER, options=option)

    zoom = "11.52249204a,151.71390185d,35y"
    url = "https://earth.google.com/web/@" + coords + "," + zoom
    driver.get(url)

    time.sleep(40)
    # driver.quit()

    driver.save_screenshot(path)

    pic = Image.open(path)
    pic = pic.rotate(90 - ang)
    pic = pic.resize((320, 180))
    pic.save(path)

    return path


if __name__ == '__main__':
    main()
