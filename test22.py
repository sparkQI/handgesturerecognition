import numpy as np

import math
import cv2



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
                #angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                s = (a + b + c) / 2
                ar = math.sqrt(s * (s - a) * (s - b) * (s - c))
                d = (2 * ar) / a
                print('-------')
                print(d)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
                print('angle')
                print(angle)
                if (angle <= math.pi / 2):  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    cv2.circle(drawing, far, 8, [211, 84, 0], -1)
            return True, cnt
    return False, 0
def f(x): return x


# main program start here
cap = cv2.VideoCapture(0)

cap_region_x_begin = 0.5  # start point/total width
cap_region_y_end = 0.8  # start point/total width

cv2.namedWindow('image')
#cv2.createTrackbar('median_size','image',0,50,f)
cv2.createTrackbar('min_val','image',0,255,f)
cv2.createTrackbar('max_val','image',0,255,f)

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    print(type(frame))

    min_val = cv2.getTrackbarPos('min_val','image')
    max_val = cv2.getTrackbarPos('max_val','image')
    median_size = cv2.getTrackbarPos('median_size','image')
    #print(frame.shape[0])
    #cv2.circle(frame, (0,0), 10, (255, 0, 0));
    #cv2.circle(frame, (640, 480), 10, ( 0,255, 0));
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                 (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (0, 255, 0), 1)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    cv2.imwrite('a.jpg',frame)
    #cv2.waitKey(0)
    # crop the image for progress:
    raw_frame = frame[0:int(cap_region_y_end * frame.shape[0]),
                int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]
    cv2.imshow('crop_frame', raw_frame)
    raw_frame_copy = raw_frame
    rows, cols, dim = raw_frame.shape
    print(raw_frame.shape[0]) #384
    print(raw_frame.shape[1]) #320
    raw_frame_copy = cv2.medianBlur(raw_frame_copy, 11)
    # cv2.imshow('median', result)
    raw_frame_copy = cv2.GaussianBlur(raw_frame_copy,(5,5),1.5)
    img = raw_frame_copy
    rows, cols, channels = img.shape


    # convert color space from rgb to ycbcr
    imgYcc = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    # convert color space from bgr to rgb
    # prepare an empty image space
    imgSkin = np.zeros(img.shape, np.uint8)
    # copy original image
    imgSkin = img.copy()

    binary = np.zeros([rows, cols])
    print(binary.shape)
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
    for r in range(rows):
        for c in range(cols):

            skin = 0

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
            # skin color detection
            if Cb > 77 and Cb < 127 and Cr > 133 and Cr < 173:
                skin = 1

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
    print(imgSkin.shape)
    binary = binary.astype(np.uint8)
    gray = cv2.cvtColor(imgSkin.copy(), cv2.COLOR_BGR2GRAY)
    result = cv2.medianBlur(binary, 11)
    # cv2.imshow('median', result)
    #result2 = cv2.GaussianBlur(result, (5, 5), 1.5)
    # cv2.imshow('gaussian', result2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(result, kernel, iterations=2)
    # opened = cv2.morphologyEx(result2,cv2.MORPH_OPEN, kernel)

    ret, bina = cv2.threshold(dilated, 127, 255, cv2.THRESH_BINARY)

    cv2.imshow('wwwwwwwwww', bina)
    print(bina.dtype)
    _, contours, _ = cv2.findContours(bina, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))

    cv2.drawContours(img, contours, -1, (0, 0, 255), 2)

    # build convexhull
    if len(contours)!=0:
        cnt = max(contours, key=lambda x: cv2.contourArea(x))
        hull = cv2.convexHull(cnt)
        cv2.drawContours(img, [hull], 0, (0, 255, 0), 2)
        isFinishCal, num = calculateFingers(cnt, img)
        number = str(num+1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, number, (120, 120), font, 2, (255, 0, 255), 2, cv2.LINE_AA)
    # defects = cv2.convexityDefects(approx, hull)
        print(num)
    # print(cnt.shape)
    # epsilon = 0.0005 * cv2.arcLength(cnt, True)
    # print(epsilon)
    # approx = cv2.approxPolyDP(cnt, epsilon, True)
    # print(approx.shape)
    #hull = cv2.convexHull(cnt)
    # defects = cv2.convexityDefects(approx, hull)
    #print(hull.shape)
    # print(defects.shape)
    # number = str(num)
    # #cv2.drawContours(img, [hull], 0, (0, 255, 0), 2)
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(img, 'num', (20, 20), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Edges", img)

    #cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
