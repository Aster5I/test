import os
import cv2
import numpy as np
import easyocr
from utility.utils import util

def recognize_license_plate(img_path):
    import os
    import cv2
    import numpy as np
    import easyocr
    from utility.utils import util
    # ...
    # define constants
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_cfg_path = os.path.join(current_dir, 'data', 'cfg', 'darknet-yolov3.cfg')
    model_weights_path = os.path.join(current_dir, 'data','model.weights')
    class_names = 'License Plate'

    # load model
    net = cv2.dnn.readNetFromDarknet(model_cfg_path, model_weights_path)

    # load image
    imge = cv2.imread(img_path)
    scale_percent = 150
    width = int(imge.shape[1] * scale_percent / 100)
    height = int(imge.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(imge, dim, interpolation=cv2.INTER_AREA)

    H, W, _ = img.shape

    # convert image
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True)

    # get detections
    net.setInput(blob)

    detections = util.get_outputs(net)

    # bboxes, class_ids, confidences
    bboxes = []
    class_ids = []
    scores = []

    for detection in detections:
        # [x1, x2, x3, x4, x5, x6, ..., x85]
        bbox = detection[:4]

        xc, yc, w, h = bbox
        bbox = [int(xc * W), int(yc * H), int(w * W), int(h * H)]

        bbox_confidence = detection[4]

        class_id = np.argmax(detection[5:])
        score = np.amax(detection[5:])

        bboxes.append(bbox)
        class_ids.append(class_id)
        scores.append(score)

    # apply nms
    bboxes, class_ids, scores = util.NMS(bboxes, class_ids, scores)

    # plot
    reader = easyocr.Reader(['ru'])
    max_score = 0
    recognized_text = ''

    for bbox_, bbox in enumerate(bboxes):
        xc, yc, w, h = bbox

        license_plate = img[int(yc - (h / 2)):int(yc + (h / 2)), int(xc - (w / 2)):int(xc + (w / 2)), :].copy()

        license_plate_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
        _, license_plate_thresh = cv2.threshold(license_plate_gray, 64, 255, cv2.THRESH_BINARY_INV)

        output = reader.readtext(license_plate_thresh)

        for out in output:
            text_bbox, text, text_score = out
            if text_score > 0.4:
                recognized_text = 'Номер: ' + text
            else:
                recognized_text = 'Номер: -'

    return recognized_text