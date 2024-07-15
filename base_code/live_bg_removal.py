import cv2
from rembg import remove
from PIL import Image
import numpy as np
import io
import threading

# Global variable to store the frame
frame = None

def capture_frames(cap):
    global frame
    while True:
        ret, new_frame = cap.read()
        if not ret:
            break
        frame = new_frame

def remove_background_from_frame(frame):
    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (320, 240))

    # Convert the frame (numpy array) to a PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB))

    # Convert PIL Image to bytes
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    image_bytes = buffered.getvalue()

    # Remove the background
    output_image_data = remove(image_bytes)

    # Convert the output bytes back to a PIL Image
    output_image_pil = Image.open(io.BytesIO(output_image_data))

    # Convert the PIL Image back to a numpy array
    output_image_np = np.array(output_image_pil)

    # Convert RGB to BGR for OpenCV
    output_image = cv2.cvtColor(output_image_np, cv2.COLOR_RGB2BGR)

    # Resize back to original frame size
    output_image = cv2.resize(output_image, (frame.shape[1], frame.shape[0]))

    return output_image

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Start frame capture thread
capture_thread = threading.Thread(target=capture_frames, args=(cap,))
capture_thread.start()

while True:
    if frame is not None:
        # Remove background from the frame
        processed_frame = remove_background_from_frame(frame)

        # Display the resulting frame
        cv2.imshow('Background Removed', processed_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close the window
cap.release()
cv2.destroyAllWindows()