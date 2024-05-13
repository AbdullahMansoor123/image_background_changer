import os,sys
import cv2
from PIL import Image

from logger import logging
from exception import CustomException


import glob
import numpy as np
import streamlit as st
from ultralytics import YOLO





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
    logging.info("Processing images...")
    try:

        fg_img = img_org.copy()
        bg_img_org = cv2.imread(bg_image_path)
        bg_img_org = resize_to_match(bg_img_org, img_org)
        bg_img_org = cv2.cvtColor(bg_img_org, cv2.COLOR_BGR2RGB)

        model = YOLO("artifacts/yolov8n-seg.pt")
        cls = [0]  # Person only
        results = model.predict(img_org, conf=0.8, classes=cls)

        for result in results:
            for mask, _ in zip(result.masks.xy, result.boxes):
                points = np.int32([mask])

                cv2.fillPoly(fg_img, points, color=(0, 0, 0))
                img_mask = img_org - fg_img
                
                cv2.fillPoly(bg_img_org, points, color=(0, 0, 0))
               
                img_with_bg = bg_img_org + img_mask
                
                logging.info("New background added to the uploaded image")
        logging.info("Image saved available for download")
        img_with_bg  = cv2.cvtColor(img_with_bg,cv2.COLOR_BGR2RGB)
        cv2.imwrite("output.png",img_with_bg)
        img_with_bg  = cv2.cvtColor(img_with_bg,cv2.COLOR_RGB2BGR)
        return img_with_bg
    
    except Exception as e:
        logging.error(f"An error occurred during image processing")
        raise CustomException(e,sys)


def app():
    st.title("Background Changing App")
    logging.info("Application started")
    uploaded_file = st.file_uploader("Choose an image")
    
    if uploaded_file is not None:
        try:
            img_org = Image.open(uploaded_file)
            img_org = np.array(img_org)
            st.image(img_org, "img_org")


            bg_image_paths = sorted(glob.glob('backgrounds/*'))
            bg_dict = {os.path.basename(p): p for p in bg_image_paths}
            bg_names = bg_dict.keys()

            selected_bg = st.selectbox(
                "Select a background image",
                bg_names)

            bg_image_path = bg_dict[selected_bg]
            st.image(bg_image_path, caption=selected_bg)

            new_image= process_images(img_org, bg_image_path)
            st.image(new_image, "Image with background added")

            with open("output.png", "rb") as file:
                st.download_button(
                    label=f"Download Image",
                    data=file,
                    file_name="output.png",
                    mime="image/png",
                )
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    app()