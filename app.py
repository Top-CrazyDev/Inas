import os,cv2
# import nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
import json
import io

ROWS = 128
COLS = 128
CHANNELS = 1
TRAIN_DIR = 'IMAGE/'
resultid = 0

train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]


def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)

def main():
    with open("prompt.json", 'r') as file:
        prompt = json.load(file)
    artist = prompt.get("artist")
    dimension = prompt.get("dimension")
    color = prompt.get("color")
    random_integer = random.randint(1, 10)
    outputid = random_integer % 10
    directory_path = './IMAGE'
    image_files = [f for f in os.listdir(directory_path) if f.endswith(('.jpg', '.png'))]
    image_array_list = []
    for filename in image_files:
        file_path = os.path.join(directory_path, filename)
        image = cv2.imread(file_path)

        image_array_list.append(image)
    resized_image = cv2.resize(image_array_list[outputid], (dimension[0], dimension[1]))
    result = convert(image = resized_image, base_color = color)

    return


def convert(image = None, base_color = [127, 127, 127]):
    if len(image.shape) == 3:
        b, g, r = cv2.split(image)
    elif len(image.shape) == 4:
        b, g, r, alpha = cv2.split(image)
    else:
        return None

    # print(np.average(image))
    average_color = [np.average(b), np.average(g), np.average(r)]
    print(average_color)

    # b = np.array(b, np.float16)
    # g = np.array(g, np.float16)
    # r = np.array(r, np.float16)
    b = b.astype(np.float64)
    g = g.astype(np.float64)
    r = r.astype(np.float64)
    

    bb = b * base_color[0] / average_color[0]
    gg = g * base_color[1] / average_color[1]
    rr = r * base_color[2] / average_color[2]


    max_val = max(np.max(gg), np.max(bb), np.max(rr))
    bb = bb * 255 / max_val
    gg = gg * 255 / max_val
    rr = rr * 255 / max_val
    print(np.max(gg), np.max(bb), np.max(rr))
    # bb = np.array(bb, np.uint8)
    bb = bb.astype(np.uint8)
    gg = gg.astype(np.uint8)
    rr = rr.astype(np.uint8)
    # gg = np.array(gg, np.uint8)
    # rr = np.array(rr, np.uint8)

    result = cv2.merge([bb, gg, rr])
    global resultid
    resultid += 1
    cv2.imshow('result', result)
    cv2.waitKey(0)

    return result



def prepare_data(images):
    m = len(images)
    X = np.zeros((m, ROWS, COLS), dtype=np.uint8)
    y = np.zeros((1, m))
    for i, image_file in enumerate(images):
        X[i,:] = read_image(str(image_file))
        if 'positive' in image_file.lower():
            y[0, i] = 1
        elif 'negative' in image_file.lower():
            y[0, i] = 0
    return X, y

train_set_x, train_set_y = prepare_data(train_images)


train_set_x_flatten = train_set_x.reshape(train_set_x.shape[0], ROWS*COLS*CHANNELS).T

train_set_x = train_set_x_flatten/255





if __name__ == "__main__":
    main()