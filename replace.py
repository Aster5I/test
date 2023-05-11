# Открываем файл для чтения
#with open("C:/Users/Home/Desktop/DIPLOMA/CODE/rec/complete_dataset.csv", "r") as f:
    # Читаем все строки в список
    #lines = f.readlines()
    # Создаем новый список для обработанных строк
    # new_lines = []
    # Для каждой строки в списке
    #for line in lines:
        # Удаляем символ переноса строки в конце
        # line = line.strip()
        # Разделяем строку по запятым в список
        # parts = line.split(",")
        # Соединяем список обратно в строку с пробелом вместо второй запятой
        # line = parts[0] + "," + parts[1] + " " + parts[2] + "," + parts[3] + "," + parts[4]
        # Добавляем обработанную строку в новый список
        # new_lines.append(line)
# Открываем файл для записи
# with open("C:/Users/Home/Desktop/DIPLOMA/CODE/rec/complete_dataset1.csv", "w") as f:
    # Для каждой строки в новом списке
    # for line in new_lines:
        # Записываем строку в файл с символом переноса строки в конце
        #f.write(line + "\n")

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr
from utility.utils import util

# define constants
model_cfg_path = os.path.join('.', 'data', 'cfg', 'darknet-yolov3.cfg')
model_weights_path = os.path.join('.', 'data', 'model.weights')
class_names_path = os.path.join('.', 'data', 'class.names')

# задайте путь к выбранному изображению
img_path = 'test_img/1.jpg'

# load class names
with open(class_names_path, 'r') as f:
    class_names = [j[:-1] for j in f.readlines() if len(j) > 2]
    f.close()

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
for bbox_, bbox in enumerate(bboxes):
    xc, yc, w, h = bbox

    license_plate = img[int(yc - (h / 2)):int(yc + (h / 2)), int(xc - (w / 2)):int(xc + (w / 2)), :].copy()

    img = cv2.rectangle(img,
                        (int(xc - (w / 2)), int(yc - (h / 2))),
                        (int(xc + (w / 2)), int(yc + (h / 2))),
                        (0, 255, 0),
                        15)

    license_plate_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)

    #_, license_plate_thresh = cv2.threshold(license_plate_gray, 64, 255, cv2.THRESH_BINARY_INV)
    #license_plate_thresh = cv2.adaptiveThreshold(license_plate_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 11, 2)

    bfilter = cv2.bilateralFilter(license_plate_gray, 11, 17, 17) #Noise reduction
    license_plate_thresh = cv2.adaptiveThreshold(bfilter, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 11, 2)
    #license_plate_thresh = cv2.Canny(bfilter, 30, 200) #Edge detection

    output = reader.readtext(license_plate_thresh)

    for out in output:
        text_bbox, text, text_score = out
        if text_score > 0.1:
            print(text, text_score)


    plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    plt.figure()
    plt.imshow(cv2.cvtColor(license_plate, cv2.COLOR_BGR2RGB))

    plt.figure()
    plt.imshow(cv2.cvtColor(license_plate_gray, cv2.COLOR_BGR2RGB))

    plt.figure()
    plt.imshow(cv2.cvtColor(license_plate_thresh, cv2.COLOR_BGR2RGB))

    plt.show()
