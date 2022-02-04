import cv2
import os
from skimage.metrics import structural_similarity
import numpy as np
import copy


def get_files_names():
    all_files = list()

    rooms = os.listdir("./Images")
    for r in rooms:
        full_imgs_path = [os.path.join(f"./Images/{r}", f) for f in os.listdir(f"./Images/{r}")]
        all_files.append(full_imgs_path)
    return all_files


def compare(orginal, img):
    before = copy.deepcopy(orginal)
    after = copy.deepcopy(img)

    # Convert images to grayscale
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between two images
    (score, diff) = structural_similarity(before_gray, after_gray, full=True)
    print("Image similarity", score)

    # The diff image contains the actual image differences between the two images
    # and is represented as a floating point data type in the range [0,1]
    # so we must convert the array to 8-bit unsigned integers in the range
    # [0,255] before we can use it with OpenCV
    diff = (diff * 255).astype("uint8")

    # Threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    mask = np.zeros(before.shape, dtype='uint8')
    filled_after = after.copy()

    for c in contours:
        area = cv2.contourArea(c)
        if area > 40:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(before, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv2.rectangle(after, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv2.drawContours(mask, [c], 0, (0, 255, 0), -1)
            cv2.drawContours(filled_after, [c], 0, (0, 255, 0), -1)

    cv2.imshow('before', before)
    cv2.imshow('after', after)
    cv2.imshow('diff', diff)
    #cv2.imshow('mask', mask)
    #cv2.imshow('filled after', filled_after)
    cv2.waitKey(0)


def main(files):
    # loop for all rooms
    for room in files:
        # Choose image
        original_file = ""
        for f in room:
            if f.endswith("Reference.JPG"):
                original_file = f
                room.remove(f)  # temp
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
            # cv2.imshow(str(f.split('\\')[-1]), img)
            compare(original, img)
            # diff = cv2.absdiff(original, img)
            # cv2.imshow("toto", diff)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # end room
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main(get_files_names())
