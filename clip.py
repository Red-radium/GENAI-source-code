from PIL import Image
import requests
import cv2
import time

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32",local_files_only=True)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32",local_files_only=True)

# Turn on the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Could not open video device")

# Initialize FPS calculation
fps = 0
prev_time = time.time()

while True:
    # Capture a single frame
    ret, frame = cap.read()
    if not ret:
        raise Exception("Failed to capture image")

    # Convert the captured frame to a PIL image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    inputs = processor(text=["a photo of a harmful trash", "a photo of a kitchen trash", "a photo of a other trash", "a photo of a recyclable trash"], images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Display the frame with prediction probabilities and FPS
    cv2.putText(frame, f"Harmful: {probs[0][0]:.2f}, Kitchen: {probs[0][1]:.2f}, Other: {probs[0][2]:.2f}, Recyclable: {probs[0][3]:.2f} | FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow('Camera', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

