# ############## 根据对txt文件 写入、读取数据，绘制曲线图##############
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

config = {
    "font.family": 'Times New Roman'
}
rcParams.update(config)


def draw_three_loss():
    plt.rcParams['savefig.dpi'] = 400  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    plt.rcParams['font.size'] = 16

    fp1 = open('lossweight.txt', 'r')
    loss1 = [1]
    loss2 = [1]
    loss3 = [1]
    i = 0
    for loss in fp1:
        loss = loss.strip('\n')  # 将\n去掉
        loss = loss.split(' ')
        loss1.append(loss[0])
        loss2.append(loss[1])
        loss3.append(loss[2])
        i += 1
    fp1.close()
    loss1.pop(len(loss1) - 1)
    loss2.pop(len(loss2) - 1)
    loss3.pop(len(loss3) - 1)

    loss1 = np.array(loss1, dtype=float)  # 将其转换成numpy的数组，并定义数据类型为float
    loss2 = np.array(loss2, dtype=float)  # 将其转换成numpy的数组，并定义数据类型为float
    loss3 = np.array(loss3, dtype=float)  # 将其转换成numpy的数组，并定义数据类型为float

    fig, ax = plt.subplots()  # 创建图实例
    x = np.linspace(0, i - 1, i)  # 创建x的取值范围
    y1 = loss1
    ax.plot(x, y1, label=chr(945))  # 作y1 = x 图，并标记此线名为linear
    y2 = loss2
    ax.plot(x, y2, label=chr(946))  # 作y2 = x^2 图，并标记此线名为quadratic
    y3 = loss3
    ax.plot(x, y3, label=chr(947))  # 作y3 = x^3 图，并标记此线名为cubic
    ax.set_xlabel('epoch')  # 设置x轴名称 x label
    ax.set_ylabel('loss weight')  # 设置y轴名称 y label
    # ax.set_title('loss weight curve of alpha, beta, gamma')  # 设置图名为Simple Plot
    ax.legend()  # 自动检测要在图例中显示的元素，并且显示
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig("lossweight.png", dpi=300)
    plt.show()  # 图形可视化


def draw_path():
    path = "74551"
    file_path = os.listdir(path)
    file_path.sort()
    lats = []
    lons = []

    for i in range(0, len(file_path)):
        file = file_path[i]
        file = file[:-4].split(',')

        lats.append(float(file[0]))
        lons.append(float(file[1]))

    plt.rcParams['savefig.dpi'] = 400  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    plt.rcParams['font.size'] = 16

    fig, ax = plt.subplots()  # 创建图实例
    x = lats
    y1 = lons
    ax.plot(x, y1, label=chr(945))  # 作y1 = x 图，并标记此线名为linear

    ax.set_xlabel('latitude')  # 设置x轴名称 x label
    ax.set_ylabel('longitude')  # 设置y轴名称 y label
    # ax.set_title('loss weight curve of alpha, beta, gamma')  # 设置图名为Simple Plot
    ax.legend()  # 自动检测要在图例中显示的元素，并且显示
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig("path.png", dpi=300)
    plt.show()  # 图形可视化


if __name__ == "__main__":
    draw_path()
