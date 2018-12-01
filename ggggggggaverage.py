# -*- coding: UTF-8 -*-

import cv2
import math
import numpy as np
import os

if __name__ == '__main__':
    # 画像の読み込み
    img_src1 = cv2.imread("./image/dambo1.jpg", 1)
    img_src2 = cv2.imread("./image/dambo2.jpg", 1)

    # 背景画像との差分を算出
    img_diff = cv2.absdiff(img_src2, img_src1)

    # 差分を二値化
    img_diffm = cv2.threshold(img_diff, 20, 255, cv2.THRESH_BINARY)[1]

    # 膨張処理、収縮処理を施してマスク画像を生成
    operator = np.ones((3, 3), np.uint8)
    img_dilate = cv2.dilate(img_diffm, operator, iterations=4)
    img_mask = cv2.erode(img_dilate, operator, iterations=4)

    # マスク画像を使って対象を切り出す
    img_dst = cv2.bitwise_and(img_src2, img_mask)

    # 表示
    cv2.imshow("Show BACKGROUND SUBSTRACTION Image", img_dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()