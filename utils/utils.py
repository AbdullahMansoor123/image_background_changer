import os,sys
from logging import Logger
from src.exception import CustomException
from ultralytics import YOLO
import cv2
import numpy as np





def resize_to_match(img_to_resize, reference_image):
    """
    Resize image_to_resize to match the size of reference_image
    """
    resized_image = cv2.resize(img_to_resize, (reference_image.shape[1], reference_image.shape[0]))
    return resized_image

def process_images(img_org, bg_image_path):
    """
    Process the given person and background images to remove person from background.

    Parameters:
    img_org (numpy.ndarray): Array representation of the uploaded image.
    bg_image_path (str): Path to the background image.

    Returns:
    numpy.ndarray: Image with person removed from the background.
    """
    Logger.info("Processing images...")
    try:
        bg_img_org = cv2.imread(bg_image_path)
        bg_img_org = resize_to_match(bg_img_org, img_org)
        bg_img = bg_img_org.copy()

        model = YOLO("artifacts/yolov8n-seg.pt")
        cls = [0]  # Person only
        results = model.predict(img_org, conf=0.8, classes=cls)

        for result in results:
            for mask, _ in zip(result.masks.xy, result.boxes):
                points = np.int32([mask])

                cv2.fillPoly(img_org, points, color=(0, 0, 0))
                new_img = img_org - img_org
                cv2.fillPoly(bg_img, points, color=(0, 0, 0))
                img_with_bg = bg_img + new_img

        Logger.info("Image processing completed.")
        return cv2.cvtColor(img_with_bg, cv2.COLOR_BGR2RGB)
    except Exception as e:
        Logger.error(f"An error occurred during image processing")
        raise CustomException(e,sys)