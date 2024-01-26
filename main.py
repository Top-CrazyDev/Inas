import cv2
import numpy as np
from tkinter.filedialog import askopenfilename

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
    cv2.imshow('result', result)
    cv2.waitKey(0)

    return result



def _main():
    base_color = [ 0, 255, 255]
    # image = cv2.imread(askopenfilename())
    image = cv2.imread(r'IMAGE/data-2.jpg')

    result = convert(image = image, base_color = base_color)

    return

if __name__ == "__main__":
    _main()