from PIL import Image, ImageFont, ImageDraw

import cv2
import random
import numpy as np
import pytest


# 盗蓝大佬的高清图像生成 https://zhuanlan.zhihu.com/p/56033037
# OpenCV-Python教程:55.OpenCV里的K-Means聚类   https://www.jianshu.com/p/b24086aab3fc
# KMeans聚类算法详解 https://zhuanlan.zhihu.com/p/184686598
def img2strimg(frame, K=5):
    """
    利用 聚类 将像素信息聚为3或5类，颜色最深的一类用数字密集地表示，阴影的一类用“-”横杠表示，明亮部分空白表示。
    ---------------------------------
    frame：需要传入的图片信息。可以是opencv的cv2.imread()得到的数组，也可以是Pillow的Image.read()。
    K：聚类数量，推荐的K为3或5。根据经验，3或5时可以较为优秀地处理很多图像了。若默认的K=5无法很好地表现原图，请修改为3进行尝试。若依然无法很好地表现原图，请换图尝试。 （ -_-|| ）
    ---------------------------------
    聚类数目理论可以取大于等于3的任意整数。但水平有限，无法自动判断当生成的字符画可以更好地表现原图细节时，“黑暗”、“阴影”、”明亮“之间边界在哪。所以说由于无法有效利用更大的聚类数量，那么便先简单地限制聚类数目为3和5。
    """
    if type(frame) != np.ndarray:
        frame = np.array(frame)

    height, width, *_ = frame.shape  # 有时返回两个值，有时三个值
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_array = np.float32(frame_gray.reshape(-1))

    # 设置相关参数。
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    # 得到labels(类别)、centroids(矩心)。
    # 如第一行6个像素labels=[0,2,2,1,2,0],则意味着6个像素分别对应着 第1个矩心、第3个矩心、第3、2、3、1个矩心。
    compactness, labels, centroids = cv2.kmeans(frame_array, K, None, criteria, 10, flags)
    centroids = np.uint8(centroids)

    # labels的数个矩心以随机顺序排列，所以需要简单处理矩心.
    centroids = centroids.flatten()
    centroids_sorted = sorted(centroids)
    # 获得不同centroids的明暗程度，0最暗
    centroids_index = np.array([centroids_sorted.index(value) for value in centroids])

    bright = [abs((3 * i - 2 * K) / (3 * K)) for i in range(1, 1 + K)]
    bright_bound = bright.index(np.min(bright))
    shadow = [abs((3 * i - K) / (3 * K)) for i in range(1, 1 + K)]
    shadow_bound = shadow.index(np.min(shadow))

    labels = labels.flatten()
    # 将labels转变为实际的明暗程度列表，0最暗。
    labels = centroids_index[labels]
    # 列表解析，每2*2个像素挑选出一个，组成（height*width*灰）数组。
    labels_picked = [labels[rows * width:(rows + 1) * width:2] for rows in range(0, height, 2)]

    canvas = np.zeros((3 * height, 3 * width, 3), np.uint8)
    canvas.fill(255)  # 创建长宽为原图三倍的白色画布。

    # 因为 字体大小为0.45时，每个数字占6*6个像素，而白底画布为原图三倍
    # 所以 需要原图中每2*2个像素中挑取一个，在白底画布中由6*6像素大小的数字表示这个像素信息。
    y = 8
    for rows in labels_picked:
        x = 0
        for cols in rows:
            if cols <= shadow_bound:
                cv2.putText(canvas, str(random.randint(2, 9)),
                            (x, y), cv2.FONT_HERSHEY_PLAIN, 0.45, 1)
            elif cols <= bright_bound:
                cv2.putText(canvas, "-", (x, y),
                            cv2.FONT_HERSHEY_PLAIN, 0.4, 0, 1)
            x += 6
        y += 6

    return canvas


# 根据盗蓝大佬的代码修改生成彩色高清字符画图片
def img2clorfultxt(frame, K=5):
    """
    利用 聚类 将像素信息聚为3或5类，颜色最深的一类用数字密集地表示，阴影的一类用“*”表示，明亮部分空白或者"-"横杠表示。
    ---------------------------------
    frame：需要传入的图片信息。可以是opencv的cv2.imread()得到的数组，也可以是Pillow的Image.read()。
    K：聚类数量，推荐的K为3或5。根据经验，3或5时可以较为优秀地处理很多图像了。若默认的K=5无法很好地表现原图，请修改为3进行尝试。若依然无法很好地表现原图，请换图尝试。 （ -_-|| ）
    ---------------------------------
    聚类数目理论可以取大于等于3的任意整数。但水平有限，无法自动判断当生成的字符画可以更好地表现原图细节时，“黑暗”、“阴影”、”明亮“之间边界在哪。所以说由于无法有效利用更大的聚类数量，那么便先简单地限制聚类数目为3和5。
    """
    if type(frame) != np.ndarray:
        frame = np.array(frame)

    height, width, *_ = frame.shape  # 有时返回两个值，有时三个值
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_array = np.float32(frame_gray.reshape(-1))

    # 设置相关参数。criteria: 这是迭代终止准则。当满足这个准则时，算法迭代停止
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    # 得到labels(类别)、centroids(矩心)。
    # 如第一行6个像素labels=[0,2,2,1,2,0],则意味着6个像素分别对应着 第1个矩心、第3个矩心、第3、2、3、1个矩心。
    compactness, labels, centroids = cv2.kmeans(frame_array, K, None, criteria, 10, flags)
    centroids = np.uint8(centroids)

    # labels的数个矩心以随机顺序排列，所以需要简单处理矩心.
    centroids = centroids.flatten()
    centroids_sorted = sorted(centroids)
    # 获得不同centroids的明暗程度，0最暗
    centroids_index = np.array([centroids_sorted.index(value) for value in centroids])

    bright = [abs((3 * i - 2 * K) / (3 * K)) for i in range(1, 1 + K)]
    bright_bound = bright.index(np.min(bright))
    shadow = [abs((3 * i - K) / (3 * K)) for i in range(1, 1 + K)]
    shadow_bound = shadow.index(np.min(shadow))

    labels = labels.flatten()
    # 将labels转变为实际的明暗程度列表，0最暗。
    labels = centroids_index[labels]
    # 列表解析，每2*2个像素挑选出一个，组成（height*width*灰）三维数组。
    labels_picked = [labels[rows * width:(rows + 1) * width:2] for rows in range(0, height, 2)]

    canvas = np.zeros((3 * height, 3 * width, 3), np.uint8)
    canvas.fill(255)  # 创建长宽为原图三倍的白色画布。

    # 因为 字体大小为0.45时，每个数字占6*6个像素，而白底画布为原图三倍
    # 所以 需要原图中每2*2个像素中挑取一个，在白底画布中由6*6像素大小的数字表示这个像素信息。cv2.FONT_HERSHEY_PLAIN为字体
    y = 8
    for rows in labels_picked:
        x = 0
        for cols in rows:

            if cols <= shadow_bound:
                # cv2.putText各参数依次是：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
                # (x,y)坐标颜色对应frame[((y // 3) - 1), ((x // 3) - 1)], 各个颜色值必须迭代强转成int组成的元组例如 (147, 132, 132)
                #  迭代转成元组的方法, tuple([int(c) for c in frame[((y // 3) - 1), ((x // 3) - 1)]]),
                #  不强制转换会报错 - Scalar value for argument 'color' is not numeric
                cv2.putText(canvas, str(random.randint(2, 9)),
                            (x, y), cv2.FONT_HERSHEY_PLAIN, 0.45,
                            tuple([int(c) for c in frame[((y // 3) - 1), ((x // 3) - 1)]]), 1)


            elif cols <= bright_bound:
                cv2.putText(canvas, "*", (x, y),
                            cv2.FONT_HERSHEY_PLAIN, 0.4, tuple([int(c) for c in frame[((y // 3) - 1), ((x // 3) - 1)]]),
                            1)
            else:
                cv2.putText(canvas, "-", (x, y),
                            cv2.FONT_HERSHEY_PLAIN, 0.4, tuple([int(c) for c in frame[((y // 3) - 1), ((x // 3) - 1)]]),
                            1)
            x += 6
        y += 6
    return canvas


# 什么鸢大佬的根据灰度重新画字符画图片, 该方法只能显示轮廓和颜色,人物比较模糊
# https://www.bilibili.com/video/BV1sQ4y1Y7P2/
def frame2text(png, save_path):
    IMG = png
    # 字符列表,可自定义文字种类和数量
    ascii_char = list('1010')

    def get_char(r, g, b, alpha=256):
        # alpha透明度
        if alpha == 0:
            return ' '
        length = len(ascii_char)
        # 计算灰度
        gray = int(0.2126 * r + 0.7152 * g + 0.0722 * b)
        unit = (256.0 + 1) / length
        # 不同灰度对应不同字符
        return ascii_char[int(gray / unit)]

    im = Image.open(IMG)
    # 宽度调整为原图1/6,由于字体宽度
    WIDTH = int(im.width / 6)
    # 高度调整为原图1/15,由于字体宽度
    HEIGHT = int(im.height / 15)
    # 黑色背景图片
    im_text = Image.new("RGB", (im.width, im.height), (255, 255, 255))
    # Image.NEAREST(最近邻采样),选择输入图像的最近像素；忽略其它输入像素.
    im = im.resize((WIDTH, HEIGHT), Image.NEAREST)

    txt = ""
    colors = []
    for i in range(HEIGHT):
        for j in range(WIDTH):
            pixel = im.getpixel((j, i))
            # 记录像素颜色信息
            colors.append((pixel[0], pixel[1], pixel[2]))
            if (len(pixel) == 4):
                txt += get_char(pixel[0], pixel[1], pixel[2], pixel[3])
            else:
                txt += get_char(pixel[0], pixel[1], pixel[2])
        txt += '\n'
        colors.append((255, 255, 255))

    # 开始画图
    dr = ImageDraw.Draw(im_text)
    # 获取默认字体
    font = ImageFont.load_default()
    x = y = 0
    # 获取字体宽高
    font_w, font_h = font.getsize(txt[1])
    # 调整比例
    font_h *= 1.37
    # 为每个ascii码字符上色
    for i in range(len(txt)):
        # 对换行符特殊处理,不上色并且保证位置正确
        if (txt[i] == '\n'):
            x += font_h
            y = -font_w
        dr.text([y, x], txt[i], colors[i])
        y += font_w
    # 保存生成的每帧彩色字符图
    im_text.save(save_path)


# 测试代码部分

def test_img2strimg():
    fp1 = "test/59.png"
    fp2 = "test/97.png"
    img = cv2.imread(fp1)
    img2 = cv2.imread(fp2)
    # 若字符画结果不好，可以尝试更改K为3。若依然无法很好地表现原图，请换图尝试。 -_-||
    str_img1 = img2strimg(img)
    str_img2 = img2strimg(img2)
    cv2.imwrite("test/59-img2strimg.png", str_img1)
    cv2.imwrite("test/97-img2strimg.png", str_img2)

def test_img2clorfultxt():
    fp1 = "test/59.png"
    fp2 = "test/97.png"
    img = cv2.imread(fp1)
    img2 = cv2.imread(fp2)
    str_img = img2clorfultxt(img)
    str_img2 = img2clorfultxt(img2)
    cv2.imwrite("test/59-img2clorfultxt.png", str_img)
    cv2.imwrite("test/97-img2clorfultxt.png", str_img2)

def test_frame2text():
    png1 = "test/59.png"
    png2 = "test/97.png"
    save_path1 = "test/59-frame2text.png"
    save_path2 = "test/97-frame2text.png"
    frame2text(png1, save_path1)
    frame2text(png2, save_path2)


if __name__ == '__main__':
    print(np.array_split(1417, 3))