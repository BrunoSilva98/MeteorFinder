import cv2
import yolo

net = yolo.create_network("config/yolov4-tiny_best.weights", "config/yolov4-tiny.cfg")