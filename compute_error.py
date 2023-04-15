import math
import os


def compute_error():
    path1 = "74551"
    file_path1 = os.listdir(path1)
    file_path1 = sorted(file_path1, key=lambda x: os.path.getctime(os.path.join(path1, x)))
    res1 = []

    for i in range(0, len(file_path1)):
        file1 = file_path1[i]
        file1 = file1[:-4].split(',')
        res1.append((float(file1[0]), float(file1[1])))

    path2 = "../sequence/processOrder/order/74551"
    file_path2 = os.listdir(path2)
    file_path2.sort()
    res2 = []

    for i in range(0, len(file_path2)):
        file2 = file_path2[i]
        # 纬度
        lat_index = file2.find("lat")
        # 高度
        alt_index = file2.find("alt")
        # 经度
        lon_index = file2.find("lon")

        start = file2[0: lat_index - 1]
        lat_pos = file2[lat_index + 4: alt_index - 1]
        alt_pos = file2[alt_index + 4: lon_index - 1]
        lon_pos = file2[lon_index + 4: -4]

        res2.append((float(lat_pos), float(lon_pos)))

    diff = []
    for i in range(0, len(res1)):
        diff.append(math.inf)

    for i in range(0, len(res1)):
        for j in range(0, len(res2)):
            lat_diff = (res1[i][0] - res2[j][0]) * 111000
            lon_diff = (res1[i][1] - res2[j][1]) * 111000 * math.cos(math.pi / 6)
            diff1 = math.sqrt(lat_diff * lat_diff + lon_diff * lon_diff)
            if diff1 < diff[i]:
                diff[i] = diff1

    print(len(diff))
    print(diff)

    for i in range(2, 50, 20):
        # 2,22,42
        ans = 0
        diff2 = 0
        for j in range(len(diff) - 1 - (i-2)//20*50, len(diff) - 1 - (i-2)//20*50 - 5 * 10, -5):
            print(i)
            print(j)
            ans += diff[j]
            lat_diff = (res1[j][0] - res1[i][0]) * 111000
            lon_diff = (res1[j][1] - res1[i][1]) * 111000 * math.cos(math.pi / 6)
            diff1 = math.sqrt(lat_diff * lat_diff + lon_diff * lon_diff)
            diff2 += diff1

        ans /= 10
        diff2 /= 10
        print(ans)
        print(diff2)

    lat_diff = (res1[-1][0] - res1[2][0]) * 111000
    lon_diff = (res1[-1][1] - res1[2][1]) * 111000 * math.cos(math.pi / 6)
    diff1 = math.sqrt(lat_diff * lat_diff + lon_diff * lon_diff)
    print(diff1)
    print(diff[-1])


if __name__ == "__main__":
    compute_error()