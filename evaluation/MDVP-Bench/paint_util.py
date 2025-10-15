import base64
import os

import cv2
import numpy as np


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def paint_text_point(image_path, points):
    image = cv2.imread(image_path)
    image_name = image_path.split("/")[-1].split(".")[0] + ".jpg"
    h, w, channels = image.shape

    # 创建一个与原始图像大小相同的黑色图像 (所有像素值为0)
    pre_alpha_image = np.zeros_like(
        image
    )  # 设置混合参数，alpha 为原图的权重，beta 为黑色图像的权重
    alpha = 0.7
    beta = 1.0 - alpha
    image = cv2.addWeighted(image, alpha, pre_alpha_image, beta, 0)

    for i, [x, y] in enumerate(points, start=1):
        # 画点
        cv2.circle(
            image, (x, y), 4, (0, 255, 0), -1
        )  # 5是圆点的半径，(0, 255, 0)是颜色（绿色），-1表示填充

        # 初始文本位置
        text_x, text_y = x + 5, y - 5
        # 调整文本位置以防止出界
        if text_x + 20 > w:  # 如果文本超出右边界
            text_x = x - 20
        if text_y - 10 < 0:  # 如果文本超出上边界
            text_y = y + 20
        if y + 10 > h:  # 如果文本超出下边界
            text_y = y - 20

        thickness = 2
        ### 红色字体
        # cv2.putText(image, str(i), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), thickness + 2)
        # # 在调整后的位置上标上数字
        # cv2.putText(image, str(i), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), thickness)

        ### 黑底白字
        # 计算文本的宽度和高度
        text = str(i)
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, thickness
        )
        # 计算矩形背景的顶点坐标
        top_left = (text_x, text_y - text_height - baseline)
        bottom_right = (text_x + text_width, text_y + baseline)
        # 绘制黑色矩形作为背景
        cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), -1)
        # 在黑色矩形上绘制白色文字
        cv2.putText(
            image,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            thickness,
        )

    save_path = os.path.join("debug", f"point_{image_name}")

    cv2.imwrite(save_path, image)

    return save_path


def paint_text_box(image_path, bbox, rgb=(0, 255, 0), rect_thickness=2):
    image = cv2.imread(image_path)
    image_name = image_path.split("/")[-1].split(".")[0] + ".jpg"
    h, w, channels = image.shape

    # 创建一个与原始图像大小相同的黑色图像 (所有像素值为0)
    pre_alpha_image = np.zeros_like(image)
    alpha = 0.8
    beta = 1.0 - alpha
    image = cv2.addWeighted(image, alpha, pre_alpha_image, beta, 0)

    for i, (x, y, box_w, box_h) in enumerate(bbox, start=1):
        # 画矩形框
        x, y, box_w, box_h = int(x), int(y), int(box_w), int(box_h)
        cv2.rectangle(image, (x, y), (x + box_w, y + box_h), rgb, rect_thickness)

        # 初始文本位置
        text_x, text_y = x + 4, y + 20
        # 调整文本位置以防止出界
        if text_x < 0:  # 如果文本超出左边界
            text_x = 0
        if text_y < 0:  # 如果文本超出上边界
            text_y = y + box_h + 15
        if text_y > h:  # 如果文本超出下边界
            text_y = h - 5

        thickness = 2
        # 获取文本宽度和高度
        text = str(i)
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, thickness
        )
        # 计算文本位置
        text_x = x + 4
        text_y = y + 20
        # 绘制文本矩形背景
        cv2.rectangle(
            image,
            (text_x, text_y - text_height - baseline),
            (text_x + text_width, text_y + baseline),
            (0, 0, 0),
            -1,
        )
        # 绘制文本
        cv2.putText(
            image,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            thickness,
        )

    save_path = os.path.join("debug", f"box_{image_name}")
    # 保存图像
    cv2.imwrite(save_path, image)

    return save_path


def paint_text_polygan(image_path, dataset, polygons, dict_id, rgb, rect_thickness):
    image = cv2.imread(image_path)
    image_name = str(dict_id) + ".jpg"
    h, w, channels = image.shape

    # 创建一个与原始图像大小相同的黑色图像 (所有像素值为0)
    pre_alpha_image = np.zeros_like(image)
    alpha = 0.8
    beta = 1.0 - alpha
    image = cv2.addWeighted(image, alpha, pre_alpha_image, beta, 0)

    for idx, item in enumerate(polygons, start=1):
        ori_points = []
        for i in range(0, len(item), 2):
            ori_points.append([int(item[i]), int(item[i + 1])])
        points = np.array(ori_points).reshape((-1, 1, 2))
        cv2.polylines(
            image, [points], isClosed=True, color=rgb, thickness=rect_thickness
        )

        min_y = float("inf")  # 初始化最小的 y 坐标为正无穷大
        min_x = float("inf")  # 初始化最小的 y 对应的 x 坐标为 None
        for i in range(0, len(item), 2):
            x = item[i]
            y = item[i + 1]
            if x < min_x:
                min_y = y
                min_x = x

        text_x = min_x
        text_y = min_y

        # 调整文本位置以防止出界
        if text_x + 20 > w:
            text_x = min_x - 20
        if text_x - 20 < 0:
            text_x = min_x + 20
        if text_y - 10 < 0:
            text_y = min_y + 20
        if min_y + 10 > h:
            text_y = min_y - 20

        thickness = 2
        text = str(idx)

        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, thickness
        )
        text_x, text_y, text_width, text_height, baseline = (
            int(text_x),
            int(text_y),
            int(text_width),
            int(text_height),
            int(baseline),
        )
        # 计算文本位置
        # text_y = text_y - 5 if text_y - 5 > 0 else text_y + text_height + 15
        # 绘制文本矩形背景
        cv2.rectangle(
            image,
            (text_x, text_y - text_height - baseline),
            (text_x + text_width, text_y + baseline),
            (0, 0, 0),
            -1,
        )
        # 绘制文本
        cv2.putText(
            image,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            thickness,
        )

    save_path = os.path.join("save_data/" + dataset, image_name)
    # 保存图像
    cv2.imwrite(save_path, image)

    return save_path
