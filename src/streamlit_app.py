import os,sys
from logging import Logger
from exception import CustomException
from PIL import Image
import glob
import numpy as np
import streamlit as st
from utils.utils_main import process_images


def app():
    st.title("Background Changing App")
    Logger.info("Application started.")
    uploaded_file = st.file_uploader("Choose an image")
    
    if uploaded_file is not None:
        try:
            my_img = Image.open(uploaded_file)
            img_org = np.array(my_img)

            bg_image_paths = sorted(glob.glob('backgrounds/*.jpg'))
            bg_dict = {os.path.basename(p): p for p in bg_image_paths}
            bg_names = bg_dict.keys()

            selected_bg = st.selectbox(
                "Select a background image",
                bg_names)

            bg_image_path = bg_dict[selected_bg]
            st.image(bg_image_path, caption=selected_bg)

            processed_image = process_images(img_org, bg_image_path)
            st.image(processed_image, caption="Processed Image")

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