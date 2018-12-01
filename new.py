import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
def calculateFingers(res,drawing):  # -> finished bool, cnt: finger count
    #  convexity defect
    hull = cv2.convexHull(res, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None):  # avoid crashing.   (BUG not found)

            cnt = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    cv2.circle(drawing, far, 8, [211, 84, 0], -1)
            return True, cnt
    return False, 0
################################################################################

print('Pixel Values Access')

imgFile = '1.jpg'

# load an original image  
img = cv2.imread(imgFile)
################################################################################  

print('YCbCr Skin Model')

rows, cols, channels = img.shape
################################################################################  

################################################################################  

# convert color space from rgb to ycbcr  
imgYcc = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

# convert color space from bgr to rgb

# prepare an empty image space  
imgSkin = np.zeros(img.shape, np.uint8)

# copy original image  
imgSkin = img.copy()

binary = np.zeros([rows, cols])
print(binary.shape)
################################################################################  

# define variables for skin rules  

Wcb = 46.97
Wcr = 38.76

WHCb = 14
WHCr = 10
WLCb = 23
WLCr = 20

Ymin = 16
Ymax = 235

Kl = 125
Kh = 188

WCb = 0
WCr = 0

CbCenter = 0
CrCenter = 0
################################################################################  

for r in range(rows):
    for c in range(cols):

        # non-skin area if skin equals 0, skin area otherwise          
        skin = 0

        ########################################################################  

        # color space transformation  

        # get values from ycbcr color space       
        Y = imgYcc.item(r, c, 0)
        Cr = imgYcc.item(r, c, 1)
        Cb = imgYcc.item(r, c, 2)

        if Y < Kl:
            WCr = WLCr + (Y - Ymin) * (Wcr - WLCr) / (Kl - Ymin)
            WCb = WLCb + (Y - Ymin) * (Wcb - WLCb) / (Kl - Ymin)

            CrCenter = 154 - (Kl - Y) * (154 - 144) / (Kl - Ymin)
            CbCenter = 108 + (Kl - Y) * (118 - 108) / (Kl - Ymin)

        elif Y > Kh:
            WCr = WHCr + (Y - Ymax) * (Wcr - WHCr) / (Ymax - Kh)
            WCb = WHCb + (Y - Ymax) * (Wcb - WHCb) / (Ymax - Kh)

            CrCenter = 154 + (Y - Kh) * (154 - 132) / (Ymax - Kh)
            CbCenter = 108 + (Y - Kh) * (118 - 108) / (Ymax - Kh)

        if Y < Kl or Y > Kh:
            Cr = (Cr - CrCenter) * Wcr / WCr + 154
            Cb = (Cb - CbCenter) * Wcb / WCb + 108
            ########################################################################

        # skin color detection  

        if Cb > 77 and Cb < 127 and Cr > 133 and Cr < 173:
            skin = 1
            # print 'Skin detected!'  

        if 0 == skin:
            imgSkin.itemset((r, c, 0), 0)
            imgSkin.itemset((r, c, 1), 0)
            imgSkin.itemset((r, c, 2), 0)
        else:
            binary.itemset((r, c), 255)
# display original image and skin image  
cv2.imshow('a', imgSkin)
cv2.imshow('b', img)
cv2.imshow('c', binary)
################################################################################  
print(imgSkin.shape)
binary = binary.astype(np.uint8)
gray = cv2.cvtColor(imgSkin.copy(),cv2.COLOR_BGR2GRAY)
result = cv2.medianBlur(binary, 11)
#cv2.imshow('median', result)
result2 = cv2.GaussianBlur(result, (5, 5), 1.5)
# cv2.imshow('gaussian', result2)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilated = cv2.dilate(result2, kernel, iterations=2)
# opened = cv2.morphologyEx(result2,cv2.MORPH_OPEN, kernel)

ret, bina = cv2.threshold(dilated,127,255,cv2.THRESH_BINARY)


cv2.imshow('wwwwwwwwww', bina)
print(bina.dtype)
_, contours, _ = cv2.findContours(bina, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
cv2.drawContours(img, contours, -1, (0, 0, 255), 2)


# build convexhull

res = max(contours, key=lambda x: cv2.contourArea(x))
# print(cnt.shape)
# epsilon = 0.0005 * cv2.arcLength(cnt, True)
# print(epsilon)
# approx = cv2.approxPolyDP(cnt, epsilon, True)
# print(approx.shape)
hull = cv2.convexHull(res)
isFinishCal,cnt = calculateFingers(res,img)
#defects = cv2.convexityDefects(approx, hull)
print(hull.shape)
#print(defects.shape)

cv2.drawContours(img, [hull], 0, (0, 255, 0), 2)
print(cnt)
#cv2.drawContours(img, [cnt], 0, (255, 0, 0), 2)
cv2.imshow("Edges", img)



# cv2.imshow('d', binary)
# ret, bina = cv2.threshold(binary,127,255,cv2.THRESH_BINARY)
# canny = cv2.Canny(bina, 30, 150)
# cv2.imshow('e', bina)
# print(bina.shape)
#_, contours, hierarchy = cv2.findContours(bina, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# print(gray.shape)
# ret, bina = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
# _, contours, hierarchy = cv2.findContours(bina, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
# cv2.drawContours(dilated, contours, -1, (0, 255, 0), 2)
# cv2.imshow("img",bina)
# cnt = max(contours, key=lambda x: cv2.contourArea(x))
# hull = cv2.convexHull(cnt)
plt.subplot(1, 3, 3), plt.imshow(binary), plt.title('contours'), plt.xticks([]), plt.yticks([])
plt.show()
