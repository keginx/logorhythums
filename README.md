# 效果
**整体效果**
![在这里插入图片描述](https://img-blog.csdnimg.cn/49d98f55cccc4b5ba6982c6dca9c4651.png)
局部图片放大效果
![在这里插入图片描述](https://img-blog.csdnimg.cn/c64cb9e66db448c9a43398ee449db648.png)
 视频转换后带雪花特效,凑合看吧, [视频地址](https://streamja.com/obkVn)
# 准备工作
## 安装FFmpeg
电脑上安装`ffpeg`处理视频并设置环境变量, windows可以参考[FFmpeg的安装教程](https://www.cnblogs.com/chadlas/p/15911072.html)这篇博客安装 
mac可以直接执行`brew install ffmpeg`安装
## 安装python依赖包
执行`pip3 install -r requirements.txt` 安装依赖包
requirements.txt 内容如下
```bash	
attrs==22.1.0
certifi==2022.9.24
charset-normalizer==2.1.1
decorator==4.4.2
exceptiongroup==1.0.4
ffmpeg==1.4
idna==3.4
imageio==2.22.4
imageio-ffmpeg==0.4.7
iniconfig==1.1.1
moviepy==1.0.3
numpy==1.23.5
opencv-contrib-python==4.6.0.66
opencv-python==4.6.0.66
packaging==21.3
Pillow==9.3.0
pluggy==1.0.0
proglog==0.1.10
pyparsing==3.0.9
pytest==7.2.0
python3-tk==0.0.0
requests==2.28.1
tomli==2.0.1
tqdm==4.64.1
urllib3==1.26.12
```
# 代码
**video.py完整代码如下:**
```python
import os, shutil, cv2, random
import datetime
import gevent

import numpy as np
from PIL import Image, ImageFont, ImageDraw
from tkinter import Tk, filedialog
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip

fps = int()
frames = int()


# 获取视频路径
def v_path():
    root = Tk()
    root.withdraw()
    file_path = (filedialog.askopenfilename(title="选择视频文件", filetypes=[('mp4', '*.mp4'), ('All Files', '*')]))
    return file_path


# 将视频分解为帧和音轨
def split_frames(v_path):
    # exist_ok表示目录存在时不必报错, 替代os.mkdir()
    os.makedirs("images", exist_ok=True)
    os.makedirs("music", exist_ok=True)
    music = AudioFileClip(v_path)
    music.write_audiofile("music/origin.wav")
    # 视频
    v1 = cv2.VideoCapture()
    v1.open(v_path)
    global fps, frames, i
    fps = v1.get(cv2.CAP_PROP_FPS)
    frames = v1.get(cv2.CAP_PROP_FRAME_COUNT)
    for i in range(int(frames)):
        ret, frame = v1.read()
        cv2.imwrite("images/%d.png" % i, frame)
    return


# 获取保存路径
def save_path():
    root = Tk()
    root.withdraw()
    file_path = (filedialog.asksaveasfilename(title="保存视频文件", filetypes=[('mp4', '*.mp4'), ('All Files', '*')]))
    return file_path


#  方式1: 通过什么鸢大佬的代码将每帧图片转换为彩色字符图片, 生成图片只能看清人物轮廓比较模糊
def frames2text():
    os.makedirs("text", exist_ok=True)
    for n in range(int(frames)):
        IMG = ("images/%d.png" % n)
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

        # 保存生成的每帧彩色字符图片
        save_path = ("text/%d" % n)
        save_path = save_path + '.png'
        im_text.save(save_path)

    # 删除 images目录
    # shutil.rmtree("images")


# 方式2: 通过修改盗蓝大佬的代码生成高清版本彩色图片, 每张图片大小变为3*3倍
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


# 用于保存第n张字符图片
def save_nth_strframe(n):
    # 读取每张图片
    fp = ("images/%d.png" % n)
    img = cv2.imread(fp)
    # 生成彩色字符画图片并保存
    str_img = img2clorfultxt(img)
    save_path = ("text/%d.png" % n)
    cv2.imwrite(save_path, str_img)


# 保存所有字符画图片: 方式1. 通过for循环将每帧图片转成字符画图片, 速度较慢可以通过多线程和多进程加速
def frames2hdtext():
    os.makedirs("text", exist_ok=True)
    for n in range(int(frames)):
        # 保存彩色字符图片到text路径
        save_nth_strframe(n)
    # 删除 images目录
    # shutil.rmtree("images")


# 保存所有字符画图片: 方式2. 通过多进程+多线程结合的方式加速
# 参考https://blog.csdn.net/qq_35869630/article/details/106026450
def trans_hdtext_thread(split_array):
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_list = {executor.submit(save_nth_strframe, i): i for i in split_array}
        # 传入 Future 列表，在每个 Future 完成时产出结果，此方法不阻塞。
        concurrent.futures.as_completed(future_list)


# 保存所有字符画图片: 方式3. 通过多进程+协程结合的方式加速
def trans_hdtext_coroutine(*params):
    '''
    创建协程
    '''
    spawn_list = []
    for array in params:
        for n in array:
            spawn = gevent.spawn(save_nth_strframe, n)
            spawn_list.append(spawn)

    # gevent.joinall 会阻塞当前流程，并执行所有给定的 greenlet，执行流程只会在所有 greenlet 执行完后才会继续向下走。 
    # https://blog.csdn.net/freeking101/article/details/53097420
    gevent.joinall(spawn_list)


def trans_hdtext_process_pool():
    os.makedirs("text", exist_ok=True)
    from multiprocessing import Pool, cpu_count
    NUMBER_OF_PROCESSES = cpu_count()
    pool = Pool(NUMBER_OF_PROCESSES - 1)

    # 将0到frames的数组拆分成NUMBER_OF_PROCESSES-1份子数组, 启动NUMBER_OF_PROCESSES-1个进程
    frames_array = np.arange(int(frames))

    # 多线程方式, 例如[1:9]的列表拆成三个子数组就变成了[ array([1,2,3]), array([4,5,6]), array([7,8,9]) ]
    # pool.map(trans_hdtext_thread, np.array_split(frames_array, NUMBER_OF_PROCESSES - 1))

    # 协程方式
    pool.map(trans_hdtext_coroutine, np.array_split(frames_array, NUMBER_OF_PROCESSES - 1))
    pool.close()
    pool.join()


# 帧合成新视频
def images2video():
    img_array = []
    # 按照第一张图片大小初始化
    img = cv2.imread("text/0.png")
    sp = img.shape
    img_width = sp[1]
    img_height = sp[0]

    for i in range(int(frames)):
        file_path = "text/%d" % i
        filename = file_path + ".png"
        img = cv2.imread(filename)

        if img is None:
            print(filename + "不存在")
            continue
        img_array.append(img)

    # 合成视频, fourcc 指定编码器,fps 要保存的视频的帧率
    out = cv2.VideoWriter("music/origin.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (img_width, img_height))
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    # 删除 text目录
    # shutil.rmtree("text")


# 合成视频和音轨
def merge_video_with_audio(save_path):
    # 原始视频
    origin_video = "music/origin.mp4"
    audio = "music/origin.wav"
    video_clip = VideoFileClip(origin_video)
    audio_clip = AudioFileClip(audio)

    # 输出到新视频
    video = video_clip.set_audio(audio_clip)
    # audio_codec='aac'改变下声道编码,避免mac播放器播放时没有声音
    video.write_videofile(save_path, audio_codec='aac')
    video_clip.close()
    audio_clip.close()


if __name__ == '__main__':
    # 原始视频路径
    video_path = v_path()

    # 保存路径
    save_video = save_path()

    starttime = datetime.datetime.now()
    split_frames(video_path)
    endtime = datetime.datetime.now()
    print("split_frames: ", starttime, endtime, (endtime - starttime).seconds)

    #  方式1: 通过什么鸢大佬的代码将每帧图片转换为彩色字符图片, 生成图片只能看清人物轮廓比较模糊
    # frames2text()
    #  方式2: 通过修改盗蓝大佬的代码生成高清版本彩色图片, 每张图片大小变为3*3倍,  通过for循环将每帧图片转成字符画图片,速度较慢 可以通过多线程加速
    # frames2hdtext()
    # 方式3: 通过多进程+多线程方式加速生成高清图片
    starttime = datetime.datetime.now()
    trans_hdtext_process_pool()
    endtime = datetime.datetime.now()
    print("trans_hdtext_process_pool: ", starttime, endtime, (endtime - starttime).seconds)

    starttime = datetime.datetime.now()
    images2video()
    endtime = datetime.datetime.now()
    print("images2video: ", starttime, endtime, (endtime - starttime).seconds)

    merge_video_with_audio(save_video)
    # # 删除music目录
    # shutil.rmtree("music")

```
<br>


**测试代码可用于单张图片转换成字符画图片**
**test.py代码如下** 
```python
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
```

**项目地址:**  [https://github.com/keginx/logorhythums](https://github.com/keginx/logorhythums)

项目结构
```bash
.
├── README.md
├── images            视频每一帧转换成的图片
│   ├── 0.png
│   └── 999.png
├── music             原始视频拆分成mp4视频和声音
│   ├── origin.mp4
│   └── origin.wav
├── requirements.txt  依赖
├── test              各种方法实现的字符画图片转换效果
│   ├── 230.png
│   ├── 59-frame2text.png
│   ├── 59-img2clorfultxt.png
│   ├── 59-img2strimg.png
│   ├── 59.png
│   ├── 97-frame2text.png
│   ├── 97-img2clorfultxt.png
│   ├── 97-img2strimg.png
│   └── 97.png
├── test.py         测试各种方法转字符画图片
├── test.txt
├── text            转换后的每帧字符画图片
│   ├── 0.png
│   └── 999.png
└── video.py        实现字符画视频转换
```
# 其他
## 遇到的一些问题
1. 弹窗功能库`tkinter`调用失败：`import _tkinter # If this fails your Python may not be configured for Tk、ModuleNotF`
**解决办法:** 执行`brew install python-tk@3.10 ` 安装
2. mac MoviePy 合成视频没有声音
**解决办法:** 在调用`write_videofile`函数时加上参数`audio_codec=“aac”`，改变下声道编码即可
3. `Scalar value for argument ‘color‘ is not numeric`错误
 **解决方法:** 用迭代的方式 转成需要的类型
	```bash
	 color = tuple ([int(x) for x in color])  #设置为整数
	```
## 一些优化方向
1. Sklearnex加速KMeans计算
`sklearnex`是一个`sklearn`加速包，大约可以提升几倍到千倍的速度。它是针对`Intel`平台的，只对`Intel CPU有`效，且CPU性能越好加速效果越显著	
2. 使用GPU
有很多GPU版本的`KMeans`算法包，大多数基于`cuda`或者`torch、tensorflow`的。
3. 代码中用的字符是`1234567890`,可以尝试用其他字符代替
## 一些建议
被转换视频建议选择一些小的视频不然会很耗时间

# 参考
1. [FFmpeg的安装教程](https://www.cnblogs.com/chadlas/p/15911072.html)
2. [Python小程序:视频转彩色字符动画](https://www.bilibili.com/video/BV1sQ4y1Y7P2/)
3. [弹窗功能库tkinter调用失败：import _tkinter # If this fails your Python may not be configured for Tk、ModuleNotF](https://blog.csdn.net/Zero_Muzi/article/details/120473899)
4. [mac MoviePy 合成视频没有声音](https://blog.csdn.net/enter89/article/details/109681095)
5. [python 多进程编程 之 加速kmeans算法](https://blog.csdn.net/sinat_37255539/article/details/91129597)
6. [Scikit-Learn学习笔记——k-means聚类：图像识别、色彩压缩](https://blog.csdn.net/jasonzhoujx/article/details/81942106) 
7. [python KMeans聚类加速计算一些简单技巧和坑总结](https://blog.csdn.net/mantoureganmian/article/details/127297878)	
8. [如何将Numpy加速700倍？用 CuPy 呀](https://www.jiqizhixin.com/articles/2019-08-29-8)
9. [Python 多线程+多进程简单使用教程，如何在多进程开多线程](https://blog.csdn.net/qq_35869630/article/details/106026450)