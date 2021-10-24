import cv2
import os
import numpy as np

cap = cv2.VideoCapture(0)

orb = cv2.ORB_create(fastThreshold=1, nfeatures=1000)

images_path = "ImageQuery"
images = []
images_names = []


videos_path = "VideoQuery"
videos = {}


images_list = os.listdir(images_path)
videos_list = os.listdir(videos_path)

for cl in images_list:
    imgCur = cv2.imread(f"{images_path}/{cl}")

    images.append(imgCur)

    images_names.append(os.path.splitext(cl)[0])

for i in range(2):
    target_video = cv2.VideoCapture(
        f"{videos_path}/video_{os.path.splitext(images_list[i])[0]}.mp4")

    videos[os.path.splitext(images_list[i])[0]] = target_video


def get_descriptors(images):
    descriptor_list = []
    kps_list = []
    for img in images:

        kp, des = orb.detectAndCompute(img, None)
        descriptor_list.append(des)
        kps_list.append(kp)

    return [descriptor_list, kps_list]


desList, kpList = get_descriptors(images)


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

while True:
    succ, webcam = cap.read()
    foundImg = webcam.copy()
    imgAug = webcam.copy()

    bf = cv2.BFMatcher()

    kp2, des2 = orb.detectAndCompute(webcam, None)

    _id = find_id(webcam, desList)

    hT, wT, cT = images[_id].shape

    """target_video = cv2.VideoCapture(
        f'{videos_path}/video_{images_names[_id]}.mp4')"""

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
        #_id = find_id(webcam, desList)
        # print(images_names[_id])

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

    cv2.imshow('found_image', foundImg)
    try:
        #cv2.imshow('box', bounding_box)
        cv2.putText(imgAug, f"Current Frame: {frameCounter}", (0, 50),
                    (cv2.FONT_HERSHEY_SIMPLEX), 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(imgAug, f"Total Frames: {target_video.get(cv2.CAP_PROP_FRAME_COUNT)}", (0, 80),
                    (cv2.FONT_HERSHEY_SIMPLEX), 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show augmented image
        cv2.imshow('augmented_video', imgAug)
    except:
        pass

    # Exiting program
    k = cv2.waitKey(1)
    if k == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
