import cv2
import numpy as np
import yolo
import os
import time


if not os.path.exists("detections/"):
    os.mkdir("detections")


def get_frames(cap, num):
    frames = list()
    for i in range(num):
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame[:frame.shape[0]-15, :]
        frames.append(frame)
    return frames


def display(img):
    cv2.imshow("inference", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_frames(frames):
    processed = list()
    kernel = np.ones((3, 3), np.uint8)
    for img in frames:
        frame = img.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.medianBlur(frame, 1)
        frame = cv2.threshold(frame, 80, 255, cv2.THRESH_BINARY)[1]
        frame = cv2.dilate(frame, kernel, iterations=1)
        processed.append(frame)
    return processed


def sum_frames(frames):
    if len(frames) > 0:
        kernel = np.zeros(frames[0].shape, dtype=np.uint8)
        for frame in frames:
            kernel = cv2.add(kernel, frame)
        return kernel
    return None


def add_weighted(frames):
    if len(frames) > 0:
        shape = frames[0].shape
        kernel = np.zeros(shape, dtype=np.uint8)
        for frame in frames:
            kernel = cv2.addWeighted(kernel, 0.8, frame, 0.4, 0)
        display(kernel)
        return kernel
    return None


def yolo_detect(net, img):
    detections = yolo.detect(net, img)
    filtered = yolo.filter(img, detections)
    detected, img = yolo.apply_nms(img, filtered, 0.4, 0.3)
    if detected:
        print("Meteoro detectado!\nImagem salva em detections/")
        timestmp = str(time.time())
        name = "detections/" + timestmp.split(".")[0] + timestmp.split(".")[1]
        cv2.imwrite(name, img)
    return detected


if __name__ == '__main__':
    PATH = "/home/bruno/data/meteoros/dataset/juntos/"
    net = yolo.create_network("config/yolov4-tiny_best.weights", "config/yolov4-tiny.cfg")

    videos = os.listdir(PATH)
    videos = [os.path.join(PATH, vid) for vid in videos if vid.endswith("avi")]
    background = None
    reset = 0

    for video in videos:
        cap = cv2.VideoCapture("M20190109_062703_MAD_1.avi")
        fps = cap.get(cv2.CAP_PROP_FPS)
        reset += 1

        if reset == 10:
            background = None
            reset = 0

        if background is None:
            background = get_frames(cap, int(fps/2))
            # original_background = add_weighted(background)
            # background_detection = yolo_detect(net, original_background)

        # if background_detection:
        #     background = None
        #     reset = 0
        #     continue

        processed_background = process_frames(background)
        sky_frames = get_frames(cap, int(fps*5))
        processed_sky = process_frames(sky_frames)
        summed_sky = sum_frames(processed_sky)
        summed_background = sum_frames(processed_background)
        diff = cv2.absdiff(summed_sky, summed_background)
        countours = cv2.findContours(diff, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        area = [count for count in countours if cv2.contourArea(count) > 30]

        if len(area) > 0:
            sky_detection = add_weighted(sky_frames)
            detection = yolo_detect(net, sky_detection)