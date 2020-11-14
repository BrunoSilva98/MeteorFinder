import cv2
import numpy as np


def create_network(model, config, gpu=False):
    net = cv2.dnn.readNetFromDarknet(config, model)
    if gpu:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    return net


def detect(net, image):
    layers = net.getLayerNames()
    layers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (608, 608), swapRB=True, crop=False)
    net.setInput(blob)
    return net.forward(layers)


def filter(image, detections, threshold=0.3):
    boxes = list()
    confidences = list()
    class_ids = list()
    (H, W) = image.shape[:2]
    output_layers = detections

    for output in output_layers:

        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > threshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return [boxes, confidences, class_ids]


def apply_nms(image, filtered_detections, conf, thresh):
    boxes, confidences, ids = filtered_detections
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf, thresh)
    bboxes = list()

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            bboxes.append([x, y, w, h])

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = "{}: {:.4f}".format("Meteor", confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return True, image
    return False, image
