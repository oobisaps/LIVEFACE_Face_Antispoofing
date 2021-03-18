import os
import sys
import math

import cv2
import numpy as np

import LIVEFACE_MOBILE_Tools as tl


class FaceCropper(object):
    CASCADE_PATH = "Model_Data/haarcascade_frontalface_default.xml"

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(self.CASCADE_PATH)

    def calculate_distance(self,p1,p2):

        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) 

    def generate(self, image_path, show_result):
        img = cv2.imread(image_path)
        if (img is None):
            print("Can't open image file")
            return 0

        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(img, 1.1, 3, minSize=(100, 100))
        if (faces is None):
            print('Failed to detect face')
            return 0

        if (show_result):
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)

            cv2.imshow('img', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        facecnt = len(faces)
        print("Detected faces: %d" % facecnt)

        print(type(faces))

        print(faces)


        height, width = img.shape[:2]

        central_point = (height / 2, width / 2)

        central_dist = 10000
        central_face = None

        for (x, y, w, h) in faces:
            r = max(w, h) / 2
            centerx = x + w / 2
            centery = y + h / 2
            nx = int(centerx - r)
            ny = int(centery - r)
            nr = int(r * 2)

            faceimg = img[ny:ny+nr, nx:nx+nr]

            face_distance = self.calculate_distance(p1 = central_point, p2 = (nx,ny)) 

            print(face_distance)

            if face_distance < central_dist:
                central_dist = face_distance
                central_face = cv2.resize(faceimg, (256, 256))

        cv2.imshow('img_alt', central_face)

        cv2.waitKey(0)


detector = FaceCropper()

path_1 = 'LIVEFACE_DATA/ATM_DATASET/LIVE/LIVEFACE_LOGS_STATE_14/00A0C257-2235-4802-A76D-ED7FC3343EBC_2019-05-31 13:38:37.539981.jpg'

path_2 = 'LIVEFACE_DATA/MOB_DATASET_NORMAL/FAKE/PHONE/STATE_1/2018-07-24 11.19.32.jpg'

# detector.generate(path, True)

img = cv2.imread(filename = path_2)
gray_img = cv2.cvtColor(src = img, code = cv2.COLOR_BGR2GRAY)

hsv_img = cv2.cvtColor(src = img, code = cv2.COLOR_BGR2HSV)
hsv_img_full = cv2.cvtColor(src = img, code = cv2.COLOR_BGR2HSV_FULL)

y_cr_cb_img_1 = cv2.cvtColor(src = img, code = cv2.COLOR_BGR2YCR_CB)
y_cr_cb_img_2 = cv2.cvtColor(src = img, code = cv2.COLOR_BGR2YCrCb)

lab_img_1 = cv2.cvtColor(src = img, code = cv2.COLOR_BGR2LAB)
lab_img_1 = cv2.cvtColor(src = img, code = cv2.COLOR_BGR2Lab)

test_img = np.concatenate((img,hsv_img), axis = 2)

print(test_img.shape)


row_1 = tl.concat_ver(imgs = [cv2.resize(src = img, dsize = (256, 256)), cv2.resize(src = gray_img, dsize = (256, 256))])
row_2 = tl.concat_ver(imgs = [cv2.resize(src = hsv_img, dsize = (256, 256)), cv2.resize(src = hsv_img_full, dsize = (256, 256))])
row_3 = tl.concat_ver(imgs = [cv2.resize(src = y_cr_cb_img_1, dsize = (256, 256)), cv2.resize(src = y_cr_cb_img_2, dsize = (256, 256))])
row_4 = tl.concat_ver(imgs = [cv2.resize(src = lab_img_1, dsize = (256, 256)), cv2.resize(src = lab_img_1, dsize = (256, 256))])


full_img = tl.concat_hor([row_1, row_2, row_3, row_4])


cv2.imshow('full_img', full_img)

cv2.waitKey()