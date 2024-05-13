import os
import io
from PIL import Image
import glob
from ultralytics import YOLO
import cv2
import numpy as np
import streamlit as st



def resize_to_match(img_to_resize, reference_image):
    """
    Resize image_to_resize to match the size of reference_image
    """
    resized_image = cv2.resize(img_to_resize, (reference_image.shape[1], reference_image.shape[0]))

    return resized_image


# def process_images(person_image_path, background_image_path):
    """
    Process the given person and background images to remove person from background.

    Parameters:
    person_image_path (str): Path to the image with a person.
    background_image_path (str): Path to the background image.

    Returns:
    numpy.ndarray: Image with person removed from the background.
    """

def app():
    st.title("Backgroung Changing App")


    uploaded_file = st.file_uploader("Choose an image")
    # buffer = io.BytesIO()
    if uploaded_file is not None:
        my_img = Image.open(uploaded_file)
        img_org = np.array(my_img)
        img = img_org.copy()

        bg_image_paths = sorted(glob.glob('backgrounds/*.jpg'))
        bg_dict = { os.path.basename(p):p for p in bg_image_paths}
        bg_names = bg_dict.keys()


        selected_bg = st.selectbox(
            "How would you like to be contacted?",
            bg_names)

        bg_image_path = bg_dict[selected_bg]
        # st.write(bg_image_path)
        st.image(bg_image_path, caption=selected_bg)

        bg_img_org = cv2.imread(bg_image_path)
        bg_img_org = resize_to_match(bg_img_org, img_org)
        bg_img = bg_img_org.copy()



        model = YOLO("yolov8n-seg.pt")
        cls = [0]  # Person only
        results = model.predict(img_org, conf=0.8, classes=cls)

        for result in results:
            for mask, _ in zip(result.masks.xy, result.boxes):
                points = np.int32([mask])

                cv2.fillPoly(img, points, color=(0, 0, 0))
                new_img = img_org - img
                cv2.fillPoly(bg_img, points, color=(0, 0, 0))
                img_with_bg = bg_img + new_img
                img_with_bg = cv2.cvtColor(img_with_bg,cv2.COLOR_BGR2RGB)
                cv2.imwrite("output.png",img_with_bg)
        
        
        # in the middle column
        with open("output.png", "rb") as file:
            st.download_button(
                label=f"Download **{"output"}.png**",
                data=file,  # download image from the in-memory buffer
                file_name="output.png",
                mime="image/png",
            )
        
    
        

        # st.download_button(label="Download your new image", data=img_with_bg.tobytes(), file_name='output.jpg', mime="image/jpg")

if __name__ == "__main__":
    app()