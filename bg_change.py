from ultralytics import YOLO
import cv2
import numpy as np



def resize_to_match(img_to_resize, reference_image):
    """
    Resize image_to_resize to match the size of reference_image
    """
    resized_image = cv2.resize(img_to_resize, (reference_image.shape[1], reference_image.shape[0]))

    return resized_image


def process_images(person_image_path, background_image_path):
    """
    Process the given person and background images to remove person from background.

    Parameters:
    person_image_path (str): Path to the image with a person.
    background_image_path (str): Path to the background image.

    Returns:
    numpy.ndarray: Image with person removed from the background.
    """
    img_org = cv2.imread(person_image_path)
    img = img_org.copy()

    bg_img_org = cv2.imread(background_image_path)
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

    return img_with_bg



# ##Example usage:
# person_image_path = "me.jpg"
# background_image_path = "backgrounds\eiffel_tower.png"
# result_image = process_images(person_image_path, background_image_path)
# cv2.imshow("Processed Image", result_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
    