import cv2
import mediapipe as mp
import numpy as np


def image_segmentation(fg_img, bg_img, mode,ksize, threshold):
    img = fg_img.copy()
    # Resize the background image to be the same size as the target image.
    bg_img = cv2.resize(bg_img, (img.shape[1], img.shape[0]))

    # blur the background image
    blur_bg = cv2.GaussianBlur(bg_img, ksize, 0)

    # meidpipe selfi segmentation
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    segment = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)
    # Convert to RGB.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Segment the original image.
    results = segment.process(img)

    # Convert to BGR.
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Retrieve segmentation mask from results.
    img_seg_mask = results.segmentation_mask

    # Apply a threhsold to generate a binary mask.
    binary_mask = img_seg_mask > threshold

    # Convert the mask to a 3 channel image.
    mask3d = np.dstack((binary_mask, binary_mask, binary_mask))
    if mode == 'replace':
        # Apply the mask to the original image and a new backgroud image.
        img_out = np.where(mask3d, img, bg_img)
        return img_out
    elif mode == 'blur_background':
        # Apply the mask to the original image and a new backgroud image.
        img_out = np.where(mask3d, img, blur_bg)
        return img_out
    elif mode == 'combine-blur-background':
        blur_img_out = np.where(mask3d, img, blur_bg)
        img_with_bg = np.where(mask3d, img, bg_img)
        return np.hstack([img_with_bg,blur_img_out])
    else:
        print("Invalid Mode Selection")




org_img = cv2.imread('yahya.jpeg')
org_img  = cv2.resize(org_img,(420,520))
# Read a background image.
bg_img = cv2.imread('waterfall.jpg')
ksize=(25,25)
threshold = 0.6


while True:
    if cv2.waitKey(0) == ord("q"):
        break
    elif cv2.waitKey(0) == ord("r"):
        mode = 'replace'
        output = image_segmentation(org_img, bg_img, mode, ksize, threshold)
        cv2.imshow('output', output)
    elif cv2.waitKey(0) == ord("b"):
        mode = 'blur_background'
        output = image_segmentation(org_img, bg_img, mode, ksize, threshold)
        cv2.imshow('output', output)
    elif cv2.waitKey(0) == ord("c"):
        mode = 'combine-blur-background'
        output = image_segmentation(org_img, bg_img, mode, ksize, threshold)
        cv2.imshow('output', output)
    else:
        cv2.imshow('output', org_img)

