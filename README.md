Here's a README.md file for your code:


# Background Changing App

I always wanted to change the background of my picture without uploading them online so i tried writing a code to that for me. This is a Streamlit web application for changing the background of an person image. It allows users to upload an image containing a person, select a background from predefined options, and then removes the person from the foreground image and blends it with the selected background.

## Setup
To run this application locally, follow these steps:


1. Clone this repository:

```bash
git clone https://github.com/AbdullahMansoor123/image_background_changer
```

2. Navigate to the cloned directory:

```bash
cd image_background_changer
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Download the YOLO model weights file `yolov8n-seg.pt` and place it in the project directory.

## Usage

After setting up the environment, you can run the Streamlit app using the following command:

```bash
streamlit run streamlit_app.py
```

Once the application starts, follow these steps:

1. Upload an image containing a person using the file uploader.
2. Select a background from the dropdown menu.
3. Click the "Download" button to save the processed image with the new background.

## Files and Directory Structure

- `app.py`: Python script containing the Streamlit application code.
- `requirements.txt`: File listing the Python dependencies required to run the application.
- `yolov8n-seg.pt`: YOLO model weights file for person segmentation.
- `backgrounds/`: Directory containing background images.

## Dependencies

- `ultralytics`: Python library for YOLO object detection.
- `streamlit`: Python library for creating web applications.
- `Pillow`: Python Imaging Library for image processing.
- `opencv-python`: OpenCV library for image manipulation.
- `numpy`: Library for numerical computing.

## License

[License](LICENSE)


## Streamlit App
**Click here to check my -->[Streamlit App](https://imagebackgroundchanger-wcwrrvh4ypexl4jtzd5gni.streamlit.app/)**