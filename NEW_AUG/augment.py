import cv2
import os
import numpy as np
import numpy.core
from PIL import Image
import math
from menu import Menu
from tkinter import Tk

cap = cv2.VideoCapture(0)

orb = cv2.ORB_create(fastThreshold=1, nfeatures=1000)

images_path = "ImageQuery"
images = []
images_names = []


videos_path = "VideoQuery"
videos = {}

desList = []
kpList = []


def make_directories():
    if not os.path.isdir(images_path) or not os.path.isdir(videos_path):
        os.mkdir(images_path)
        os.mkdir(videos_path)


make_directories()


def find_id(img, descriptor_list, thresh=15):
    _, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher()

    matchList = []
    flag = -1
    try:
        for des in descriptor_list:
            matches = bf.knnMatch(des, des2, k=2)

            good = [[m] for m, n in matches if m.distance < 0.75*n.distance]
            matchList.append(len(good))
    except:
        pass

    if len(matchList) != 0 and max(matchList) > thresh:
        flag = matchList.index(max(matchList))

    return flag


detection = False
frameCounter = 0


tk = Tk()
menu = Menu(tk, images_path, videos_path, images,
            images_names, videos, desList, kpList)
tk.mainloop()

while True:

    succ, webcam = cap.read()
    foundImg = webcam.copy()
    imgAug = webcam.copy()

    bf = cv2.BFMatcher()

    kp2, des2 = orb.detectAndCompute(webcam, None)

    _id = find_id(webcam, desList)

    try:
        hT, wT, cT = images[_id].shape
    except:
        pass

    target_video = videos[images_names[_id]]

    s, imgVid = target_video.read()

    try:
        imgVid = cv2.resize(imgVid, (wT, hT))
    except Exception as e:
        pass

    if not detection:
        # If image is not detected, reset video to beginning position

        target_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frameCounter = 0
    else:
        # If image is detected and frame counter is max frames of video - rest frames to 0
        if frameCounter == target_video.get(cv2.CAP_PROP_FRAME_COUNT):

            target_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0

        success, imgVid = target_video.read()

        try:
            imgVid = cv2.resize(imgVid, (wT, hT))
        except Exception as e:
            pass

    if des2 is not None:

        try:
            imgVid = cv2.resize(imgVid, (wT, hT))
        except Exception as e:
            pass

        # Find the matches of the discovered pic in the descriptors collecetd of image that
        # are in the images folder
        matches = bf.knnMatch(np.asarray(
            desList[_id], np.float32), np.asarray(des2, np.float32), k=2)

        # Find good matches
        good = [m for m, n in matches if m.distance < 0.75*n.distance]

        # Draw matches on the image
        foundImg = cv2.drawMatches(
            images[_id], kpList[_id], webcam, kp2, good, None, flags=2)

        if len(good) > 20:
            # For good matches, augment respective video onto the image

            detection = True
            srcPts = np.float32(
                [kpList[_id][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

            dstPts = np.float32(
                [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            matrix, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5)

            pts = np.float32([[0, 0], [0, hT], [wT, hT], [wT, 0]]
                             ).reshape(-1, 1, 2)

            dst = cv2.perspectiveTransform(pts, matrix)

            bounding_box = cv2.polylines(
                webcam, [np.int32(dst)], True, (255, 0, 255), 3)

            try:
                warped_image = cv2.warpPerspective(
                    imgVid, matrix, (webcam.shape[1], webcam.shape[0]))
            except Exception as e:
                pass

            maskNew = np.zeros((webcam.shape[0], webcam.shape[1]), np.uint8)
            cv2.fillPoly(maskNew, [np.int32(dst)], (255, 255, 255))
            maskInverse = cv2.bitwise_not(maskNew)
            imgAug = cv2.bitwise_and(imgAug, imgAug, mask=maskInverse)
            imgAug = cv2.bitwise_or(warped_image, imgAug)
        else:
            detection = False

        frameCounter += 1

    #cv2.imshow('found_image', foundImg)
    try:

        # Display bar
        nextFrame = target_video.get(cv2.CAP_PROP_POS_FRAMES)
        totalFrames = target_video.get(cv2.CAP_PROP_FRAME_COUNT)
        comp = nextFrame / totalFrames
        thicc = 20
        y = math.ceil(webcam.shape[1] - webcam.shape[1]/25)
        x = 0
        w = webcam.shape[0]

        vid_length = (totalFrames//25)/60
        prec_vid_length = "{:.2f}".format(vid_length)
        current_time = "{:.2f}".format(frameCounter/25)

        cv2.line(imgAug, (10, 1050),
                 (math.ceil(w*comp), 1050), (0, 0, 255), thicc)

        cv2.putText(imgAug, f"To exit, press 'q'.", (0, 1000),

                    (cv2.FONT_HERSHEY_SIMPLEX), 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show augmented image
        cv2.imshow('augmented_video', imgAug)

    except Exception as e:
        print(str(e))

    # Exiting program
    k = cv2.waitKey(1)
    if k == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
