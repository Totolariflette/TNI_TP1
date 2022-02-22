import sys

import cv2
import os
from skimage.metrics import structural_similarity
import numpy as np
import copy


def get_files_names():
    all_files = list()

    rooms = os.listdir(sys.argv[1])
    for r in rooms:
        full_imgs_path = [os.path.join(f"./Images/{r}", f) for f in os.listdir(f"./Images/{r}")]
        all_files.append(full_imgs_path)
    return all_files


def compare(original, img):
    before = copy.deepcopy(original)
    after = copy.deepcopy(img)

    # Convert images to grayscale
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    # before_gray, after_gray = equalize_dist(before_gray, after_gray)

    # difference
    (score, diff) = structural_similarity(before_gray, after_gray, full=True)
    print("Image similarity", score)

    diff = (diff * 255).astype("uint8")

    # Threshold
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    mask = np.zeros(before.shape, dtype='uint8')
    filled_after = after.copy()

    for c in contours:
        area = cv2.contourArea(c)
        if 600 < area:
            x, y, w, h = cv2.boundingRect(c)
            if 30 < x < 910 - w and 50 < y < 510:
                cv2.rectangle(before, (x, y), (x + w, y + h), (36, 255, 12), 2)     # (960, 540)
                cv2.rectangle(after, (x, y), (x + w, y + h), (36, 255, 12), 2)
                cv2.drawContours(mask, [c], 0, (0, 255, 0), -1)
                cv2.drawContours(filled_after, [c], 0, (0, 255, 0), -1)

    cv2.imshow('before', before)
    cv2.imshow('after', after)
    cv2.imshow('diff', diff)
    #cv2.imshow('mask', mask)
    #cv2.imshow('filled after', filled_after)
    cv2.waitKey(0)


def compare_test(original, img):
    before = copy.deepcopy(original)
    after = copy.deepcopy(img)

    # Convert images to grayscale
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    # Equalize
    before_gray_clahe, after_gray_clahe = equalize_dist(before_gray, after_gray)
    cv2.imshow("before_gray_clahe", before_gray_clahe)
    cv2.imshow("after_gray_clahe", after_gray_clahe)

    #thresh_test(before_gray_clahe, after_gray_clahe)

    # Thresholds
    # th = 80
    # max_val = 100
    #
    # ret, o1 = cv2.threshold(before_gray_clahe, th, max_val, cv2.THRESH_TOZERO)
    # ret, o3 = cv2.threshold(after_gray_clahe, th, max_val, cv2.THRESH_TOZERO)
    #
    # ret, thresh1 = cv2.threshold(o1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # ret, thresh2 = cv2.threshold(o3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #
    # difference = cv2.subtract(thresh1, thresh2)

    thresh1 = cv2.adaptiveThreshold(before_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 3)
    thresh2 = cv2.adaptiveThreshold(after_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 5)

    difference = cv2.subtract(thresh1, thresh2)

    thresh = cv2.threshold(difference, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    # Contours
    mask = np.zeros(thresh1.shape, dtype='uint8')
    filled_after = thresh2.copy()
    for c in contours:
        area = cv2.contourArea(c)
        if area > 500:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(before, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv2.rectangle(after, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv2.drawContours(mask, [c], 0, (0, 255, 0), -1, )
            cv2.drawContours(filled_after, [c], 0, (0, 255, 0), -1)

    cv2.imshow("thresh1", thresh1)
    cv2.imshow("thresh2", thresh2)
    cv2.imshow('before', before)
    cv2.imshow('after', after)
    cv2.imshow('difference', difference)
    #cv2.imshow('mask', mask)
    #cv2.imshow('filled after', filled_after)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def thresh_test(before_gray, after_gray):
    thresh1 = cv2.adaptiveThreshold(before_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh2 = cv2.adaptiveThreshold(before_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 3)
    thresh3 = cv2.adaptiveThreshold(before_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 5)
    thresh4 = cv2.adaptiveThreshold(before_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 4)
    thresh11 = cv2.adaptiveThreshold(after_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh21 = cv2.adaptiveThreshold(after_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 5)
    thresh31 = cv2.adaptiveThreshold(after_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
    thresh41 = cv2.adaptiveThreshold(after_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)

    final = np.concatenate((thresh1, thresh2, thresh3, thresh4), axis=1)
    final1 = np.concatenate((thresh11, thresh21, thresh31, thresh41), axis=1)
    cv2.imshow('rect.jpg', final)
    cv2.imshow('rect1.jpg', final1)


def equalize_dist(before_gray, after_gray):
    before_gray_eq = cv2.equalizeHist(before_gray)
    after_gray_eq = cv2.equalizeHist(after_gray)

    clahe = cv2.createCLAHE(clipLimit=40)
    before_gray_clahe = clahe.apply(before_gray_eq)
    after_gray_clahe = clahe.apply(after_gray_eq)

    return before_gray_clahe, after_gray_clahe


def main(files):
    # loop for all rooms
    for room in files:
        # Choose image
        original_file = ""
        for f in room:
            if f.endswith("Reference.JPG"):
                original_file = f
                room.remove(f)
                break
        original = cv2.imread(original_file)
        original = cv2.resize(original, (960, 540))

        # Display images
        cv2.imshow("Original", original)
        cv2.waitKey(0)
        # loop for each img (do treatments here)
        for f in room:
            img = cv2.imread(f)
            img = cv2.resize(img, (960, 540))

            compare(original, img)
            #test(original, img)
            #compare_test(original, img)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # end room
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main(get_files_names())
