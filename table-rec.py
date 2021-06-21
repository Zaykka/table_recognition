import sys
import re
import cv2
from pdf2image import convert_from_path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract

def convert():
    f_name = sys.argv[1]
    is_pdf = f_name[f_name.rfind('.') + 1:]
    if is_pdf == 'pdf':
        pages = convert_from_path(sys.argv[1], 500)
        for i, page in enumerate(pages):
            page.save('out' + str(i) + '.jpg', 'JPEG')
    else:
        pages = [1]

    for i in range(len(pages)):
        if is_pdf == 'pdf':
            f_name = 'out' + str(i) + '.jpg'

        grey_scale, read_image = read_file(f_name)

        combine, inverse = hor_ver_detect(read_image, grey_scale)

        cont, _ = cv2.findContours(combine, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cont, boxes = get_boxes(cont, method="top-to-bottom")

        final_box, avg = contour_pos(boxes, cont, read_image)

        order, hor, total = sort_boxes(final_box, avg)

        extract = extract_values(order, inverse)

        to_xlsx(extract, hor, total, sys.argv[1], i)


def read_file(sample):
    read_image = cv2.imread(sample, 0)
    convert_bin, grey_scale = cv2.threshold(read_image,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    grey_scale = 255 - grey_scale
    return grey_scale, read_image


def hor_ver_detect(read_image, grey_scale):
    length = np.array(read_image).shape[1]//100
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
    final = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    horizontal_detect = cv2.erode(grey_scale, horizontal_kernel, iterations=3)
    hor_line = cv2.dilate(horizontal_detect, horizontal_kernel, iterations=3)

    vertical_detect = cv2.erode(grey_scale, vertical_kernel, iterations=3)
    ver_lines = cv2.dilate(vertical_detect, vertical_kernel, iterations=3)

    combine = cv2.addWeighted(ver_lines, 0.5, hor_line, 0.5, 0.0)
    combine = cv2.erode(~combine, final, iterations=2)
    thresh, combine = cv2.threshold(combine,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    convert_xor = cv2.bitwise_xor(read_image,combine)
    inverse = cv2.bitwise_not(convert_xor)

    return combine, inverse


def get_boxes(num, method="left-to-right"):
    invert = False
    flag = 0
    if method == "right-to-left" or method == "bottom-to-top":
        invert = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        flag = 1
    boxes = [cv2.boundingRect(c) for c in num]
    (num, boxes) = zip(*sorted(zip(num, boxes),
    key=lambda b:b[1][flag], reverse=invert))
    return (num, boxes)


def contour_pos(boxes, cont, read_image):
    dim = [boxes[i][3] for i in range(len(boxes))]
    avg = np.mean(dim)

    final_box = []
    weidth = np.array(read_image).shape[1] // 2
    height = np.array(read_image).shape[0] // 10
    for c in cont:
        s1, s2, s3, s4 = cv2.boundingRect(c)
        if (s3 < weidth and s4 < height):
            final_box.append([s1,s2,s3,s4])

    return final_box, avg


def fin_order(hor, total, mid):
    order = []
    for i in range(len(hor)):
        arrange = []
        for k in range(total):
            arrange.append([])
        for j in range(len(hor[i])):
            sub = abs(mid - (hor[i][j][0] + hor[i][j][2]/4))
            lowest = min(sub)
            idx = list(sub).index(lowest)
            arrange[idx].append(hor[i][j])
        order.append(arrange)
    return order


def sort_boxes(final_box, avg):
    hor=[]
    ver=[]

    for i in range(len(final_box)):
        if i == 0:
            ver.append(final_box[i])
            last = final_box[i]
        else:
            if final_box[i][1] <= last[1] + avg/2:
                ver.append(final_box[i])
                last = final_box[i]
                if i == len(final_box) - 1:
                    hor.append(ver)
            else:
                hor.append(ver)
                ver=[]
                last = final_box[i]
                ver.append(final_box[i])
    total = mid = 0

    for i in range(len(hor)):
        t = len(hor[i])
        if t > total:
            total = t
        mid = [int(hor[i][j][0]+hor[i][j][2]/2) for j in range(len(hor[i])) if hor[0]]
    mid = np.array(mid)
    mid.sort()

    order = fin_order(hor, total, mid)
    return order, hor, total


def extract_values(order, inverse):
    extract=[]
    for i in range(len(order)):
        for j in range(len(order[i])):
            inside=''
            if(len(order[i][j])==0):
                extract.append(' ')
            else:
                for k in range(len(order[i][j])):
                    side1,side2,width,height = order[i][j][k][0],order[i][j][k][1], order[i][j][k][2],order[i][j][k][3]
                    final_extract = inverse[side2:side2+height, side1:side1+width]
                    final_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                    get_border = cv2.copyMakeBorder(final_extract,2,2,2,2, cv2.BORDER_CONSTANT,value=[255,255])
                    resize = cv2.resize(get_border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    dil = cv2.dilate(resize, final_kernel,iterations=1)
                    ero = cv2.erode(dil, final_kernel,iterations=2)
                    ocr = pytesseract.image_to_string(ero, lang='rus')
                    if(len(ocr)==0):
                        ocr = pytesseract.image_to_string(ero, lang='rus')
                    inside = inside +" "+ ocr
                    inside = re.sub("\n", ' ', inside)
                    inside = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff\n]', ' ', inside)
                extract.append(inside)
    return extract


def to_xlsx(extract, hor, total, f_name, n):
    a = np.array(extract)
    dataset = pd.DataFrame(a.reshape(len(hor), total))
    pos1 = f_name.rfind('/') + 1
    if pos1 == -1:
        pos1 = 0
    pos2 = f_name.rfind('.')
    f_name = f_name[pos1: pos2]
    if n == 0:
        with pd.ExcelWriter(f_name + '.xlsx', engine="openpyxl",
                        mode='w', sheet_name = n) as writer:
            dataset.to_excel(writer)
    else:
        with pd.ExcelWriter(f_name + '.xlsx', engine="openpyxl",
                        mode='a', sheet_name = n) as writer:
            dataset.to_excel(writer)


convert()
