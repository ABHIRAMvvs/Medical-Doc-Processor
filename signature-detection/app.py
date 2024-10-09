from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2 as cv
from models import model
import numpy as np
from io import BytesIO
import shutil

app = FastAPI()

@app.post("/detect_signatures/")
async def detect_signatures(file: UploadFile = File(...)):
    # Read image directly from the uploaded file
    file_bytes = await file.read()  # Read the file bytes
    np_img = np.frombuffer(file_bytes, np.uint8)  # Convert bytes to numpy array
    image = cv.imdecode(np_img, cv.IMREAD_COLOR)  # Decode into an image

    # Convert image to grayscale
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Convert the grayscale image to binary (black and white)
    _, binary_image = cv.threshold(gray_image, 128, 255, cv.THRESH_BINARY)

    # Convert the single-channel binary image to a 3-channel image (YOLO requires 3-channel input)
    binary_image_rgb = cv.cvtColor(binary_image, cv.COLOR_GRAY2BGR)

    # Perform object detection on the 3-channel binary image
    results = model(binary_image_rgb)
    # conf_threshold = 0.5
    # results = model(binary_image_rgb, conf=conf_threshold)


    # Prepare the detection data (just numbers and bounding boxes)
    detections = []
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Get the bounding boxes

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        detections.append({
            "detection_number": idx + 1,  # 1-based index for detection number
            "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        })

    # Return JSON response with total number of detections and detection details
    return JSONResponse({
        "total_detections": len(detections),
        "detections": detections
    })
