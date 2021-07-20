from google.cloud import vision
import io
import os
import re
import glob
import string
import json
import boto3
import csv
import cv2
import numpy as np
import math

# -*- coding: utf-8 -*-

# Configure environment for google cloud vision
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "client_secrets.json"

# Create a ImageAnnotatorClient
VisionAPIClient = vision.ImageAnnotatorClient()

path = r'C:\Users\User\Documents\pan_signature\data\pancards'

def word_segment(image, word_result, image_name):
    word_list = []
    box_list = []
    for (idx, item) in enumerate(word_result[1:]):
        word_list.append(item.description)
        box_list.append(item.bounding_poly)
        clone = image.copy()
        p1x, p1y = item.bounding_poly.vertices[0].x, item.bounding_poly.vertices[0].y
        p2x, p2y = item.bounding_poly.vertices[1].x, item.bounding_poly.vertices[1].y
        p3x, p3y = item.bounding_poly.vertices[2].x, item.bounding_poly.vertices[2].y
        p4x, p4y = item.bounding_poly.vertices[3].x, item.bounding_poly.vertices[3].y
        cnt = np.array([
            [[p1x, p1y]],
            [[p2x, p2y]],
            [[p3x, p3y]],
            [[p4x, p4y]]
        ])

        angle = np.rad2deg(np.arctan2(item.bounding_poly.vertices[2].y - item.bounding_poly.vertices[3].y,
         item.bounding_poly.vertices[2].x - item.bounding_poly.vertices[3].x))
        # print(item.description)
        # print(angle)
        x_max = max(p1x,p2x, p3x, p4x)
        x_min = min(p1x,p2x, p3x, p4x)
        y_max = max(p1y,p2y, p3y, p4y)
        y_min = min(p1y,p2y, p3y, p4y)

        cx = x_min + (x_max-x_min)/2
        cy = y_min + (y_max-y_min)/2

        height = math.sqrt((p4x-p1x) * (p4x-p1x) + (p4y-p1y) * (p4y-p1y)) * 1.1
        width = math.sqrt((p4x-p3x) * (p4x-p3x) + (p4y-p3y) * (p4y-p3y)) + height * 0.2

        rect = ((cx, cy), (width, height), angle)
        # print(rect)

        # the order of the box points: bottom left, top left, top right, bottom right
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # print("bounding box: {}".format(box))

        # get width and height of the detected rectangle
        width = int(rect[1][0])
        height = int(rect[1][1])

        src_pts = box.astype("float32")
        # corrdinate of the points in box points after the rectangle has been straightened
        dst_pts = np.array([[0, height-1],
                            [0, 0],
                            [width-1, 0],
                            [width-1, height-1]], dtype="float32")

        # the perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # directly warp the rotated rectangle to get the straightened rectangle
        warped = cv2.warpPerspective(clone, M, (width, height))

        (wt, ht) = (128, 32)
        (h, w) = (warped.shape[0], warped.shape[1])
        fx = w / wt
        fy = h / ht
        f = max(fx, fy)
        newSize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1)) # scale according to f (result at least 1 and at most wt or ht)
        img = cv2.resize(warped, newSize)
        target = np.ones([ht, wt]) * 255
        target[0:newSize[1], 0:newSize[0]] = img

        cv2.imwrite('out/{}_{}.png'.format(image_name, idx), target)
        # print(idx, image_name)
        cv2.waitKey(0)

for filename in glob.glob(os.path.join(path, '*.*')):

    with io.open(filename, 'rb') as image_file:
        content = image_file.read()

    # Send the image content to vision and stores text-related response in text
    # pylint: disable=no-member
    image = vision.types.Image(content=content)
    response = VisionAPIClient.document_text_detection(image=image, image_context={"language_hints": ["en"]})

    document = response.full_text_annotation
    word_result = response.text_annotations
    # print(document)
    image = cv2.imread(filename, 0)

    image_name = os.path.basename(filename).split('.')[0]
    word_segment(image, word_result,image_name)
