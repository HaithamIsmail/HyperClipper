import os
import re
import requests
import io
import base64
from PIL import Image
from typing import List, Union
import uuid
import cv2

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def numerical_sort(value):
    parts = re.split(r'(\d+)', value)
    return [int(text) if text.isdigit() else text for text in parts]

def get_text_embedding(text: Union[str, List[str]], embedding_url: str) -> List[List[float]]:
    """
    Send text to the CLIP model and get feature embeddings.
    
    Args:
        text (str): The text to embed
        api_url (str): Base URL of your Modal API
        
    Returns:
        List[List[float]]: Feature embeddings for the text
    """

    if isinstance(text, str):
        text = [text]
        
    endpoint = f"{embedding_url}/extract-text-features"
    
    print(text)

    payload = {
        "text": text
    }
    
    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        
        result = response.json()
        return result["features"]
        
    except requests.exceptions.RequestException as e:
        print(f"Error calling text features API: {e}")
        raise
    except KeyError as e:
        print(f"Unexpected response format: {e}")
        raise


def get_image_embedding(images: List[str], embedding_url: str) -> List[List[float]]:
    """
    Send images to the CLIP model and get feature embeddings.
    
    Args:
        images (List[str]): List of image file paths or PIL Images
        api_url (str): Base URL of your Modal API
        
    Returns:
        List[List[float]]: Feature embeddings for each image
    """
    endpoint = f"{embedding_url}/extract-image-features"
    
    # Convert images to base64 strings
    base64_images = []
    
    for img in images:
        if isinstance(img, str):
            # Assume it's a file path
            # with open(img, "rb") as image_file:
                # img_bytes = image_file.read()
            img_bytes = img
            base64_images.append(img_bytes)
        elif isinstance(img, Image.Image):
            # PIL Image object
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_bytes = buffer.getvalue()

            # Encode to base64
            base64_string = base64.b64encode(img_bytes).decode('utf-8')
            base64_images.append(base64_string)
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
    
    payload = {
        "images": base64_images
    }
    
    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        
        result = response.json()
        return result["features"]
        
    except requests.exceptions.RequestException as e:
        print(f"Error calling image features API: {e}")
        raise
    except KeyError as e:
        print(f"Unexpected response format: {e}")
        raise


def sample_from_video(video_path: str, sampling_rate=0.5) -> list[Image.Image]:
    """
    Samples from a video according to given sampling rate and returns a list of images

    Args:
        video_path (str): path to video
        sampling_rate (float): frames per second, how many frames to take from each second
                               e.g., 0.5 means take 1 frame every 2 seconds.
                               e.g., 2 means take 2 frames every 1 second.

    Returns:
        list[Image.Image]: a list of PIL images
    """
    print(f"Attempting to open video: {video_path}")
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)

    if fps == 0: # Handle cases where FPS might not be readable or is zero
        print(f"Error: Video FPS is {fps}. Cannot calculate sampling.")
        video.release()
        return []
    
    if sampling_rate <= 0:
        print(f"Error: sampling_rate ({sampling_rate}) must be positive.")
        video.release()
        return []

    # Calculate the frame interval.
    # If sampling_rate is 0.5 FPS (1 frame every 2s) and video is 30 FPS,
    # interval = 30 / 0.5 = 60. So, take frame 0, 60, 120...
    # If sampling_rate is 2 FPS (2 frames every 1s) and video is 30 FPS,
    # interval = 30 / 2 = 15. So, take frame 0, 15, 30...
    frame_interval = round(fps / sampling_rate)
    # Ensure we always advance at least one frame to avoid infinite loops if fps/sampling_rate is too small
    frame_interval = max(1, int(frame_interval))


    print(f"Video Info - Total Frames: {total_frames}, FPS: {fps:.2f}, Desired Sample Rate: {sampling_rate} fps")
    print(f"Calculated frame interval: Take 1 frame every {frame_interval} original frames.")

    current_frame_pos = 0
    images = []

    while current_frame_pos < total_frames:
        video.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)
        success, frame_bgr = video.read() # frame_bgr is a NumPy array in BGR format

        if not success:
            # This might happen if we try to seek beyond the last valid frame
            # or if there's a read error.
            print(f"Warning: Failed to read frame at position {current_frame_pos}. Ending capture.")
            break

        # Convert the BGR frame to RGB for PIL
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Create a PIL Image from the RGB NumPy array
        image = Image.fromarray(frame_rgb)

        # If you want to display/save for debugging:
        # image.show(title=f"Frame {current_frame_pos}") # Displays the image
        # image.save(f"debug_frame_{current_frame_pos}.png") # Saves the image

        images.append(image)
        # print(f"Captured frame {current_frame_pos}")

        current_frame_pos += frame_interval

    video.release()
    print(f"Successfully sampled {len(images)} images.")
    return images

def convert_base64_to_image(base64_image: str) -> Image.Image:
    """
    Convert a base64 encoded string to a PIL Image.

    Args:
        base64_image (str): The base64 encoded image.

    Returns:
        PIL.Image.Image: The decoded image.
    """
    image_data = base64.b64decode(base64_image)
    image = Image.open(io.BytesIO(image_data))
    return image

def convert_image_to_base64(image: Image.Image, image_ext: str) -> str:
    """
    Convert an image to a base64 encoded string.

    Args:
        image (PIL.Image.Image): The image to convert.
        image_ext (str): The image file extension (e.g., 'png', 'jpeg', "webp", non-animated "gif").

    Returns:
        str: The base64 encoded image.
    """
    buffered = io.BytesIO()
    image.save(buffered, format=image_ext.upper())
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_base64
