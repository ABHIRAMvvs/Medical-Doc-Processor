from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import fitz  # PyMuPDF for PDF processing
import cv2
import layoutparser as lp
import paddleocr
from PIL import Image
import numpy as np
import requests
import logging
import boto3
from transformers import TableTransformerForObjectDetection, DetrImageProcessor

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# AWS S3 Credentials
S3_ACCESS_KEY = 'd812703636ea4510a0c7651623e5a386'
S3_SECRET_KEY = '2eab9ce147cb4671b00fe9ab37ca982e'
S3_ENDPOINT = 'https://s3.sgp.io.cloud.ovh.net'
S3_BUCKET_NAME = 'abhiram-s3'
REGION = 'sgp'

# Initialize PaddleOCR
ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang='en')

# Initialize table structure recognition model
structure_model = TableTransformerForObjectDetection.from_pretrained("structural-transformer_rel_v0.1")
image_processor = DetrImageProcessor()

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    endpoint_url=S3_ENDPOINT,
    region_name=REGION
)

# External API URLs for document classification, stamp detection, and sign detection
CLASSIFICATION_API_URL = "http://127.0.0.1:8080/classify_document_type"
SIGNATURE_API_URL = "http://127.0.0.1:8081/detect_signatures/"
STAMP_API_URL = "http://127.0.0.1:8082/detect_stamps/"

# Helper function to upload image to S3
def upload_to_s3(file_content: bytes, file_name: str) -> str:
    # Prepend the folder path to the file name
    s3_key = f"vertex-view-resources/annotated-images/{file_name}"
    s3_client.put_object(Body=file_content, Bucket=S3_BUCKET_NAME, Key=s3_key)
    s3_url = f"{S3_ENDPOINT}/{S3_BUCKET_NAME}/{s3_key}"
    return s3_url

# Helper function to convert image to bytes
def image_to_bytes(image: np.ndarray) -> bytes:
    _, buffer = cv2.imencode('.jpg', image)
    return buffer.tobytes()

# Helper function for OCR text extraction
def extract_ocr_text(image: np.ndarray) -> str:
    ocr_results = ocr.ocr(image)
    text = " ".join([line[1][0] for result in ocr_results for line in result])
    return text

# Helper function to extract table data as CSV
def extract_table_data(image: np.ndarray, table_bbox) -> str:
    try:
        table_bbox = [int(coord) for coord in table_bbox]
        cropped_image = image[table_bbox[1]:table_bbox[3], table_bbox[0]:table_bbox[2]]
        ocr_results = ocr.ocr(cropped_image)
        csv_rows = [",".join([cell[1][0] for cell in row]) for row in ocr_results]
        return "\n".join(csv_rows)
    except Exception as e:
        logger.error(f"Error extracting table data: {e}")
        return ""

# Helper function for PDF to Image conversion
def pdf_to_images(pdf: UploadFile) -> list:
    pdf_document = fitz.open(stream=pdf.file.read(), filetype="pdf")
    return [
        np.array(Image.frombytes("RGB", [page.get_pixmap().width, page.get_pixmap().height], page.get_pixmap().samples))
        for page in pdf_document
    ]

@app.post("/process_pdf/")
async def process_pdf(pdf: UploadFile = File(...)):
    images = pdf_to_images(pdf)
    output_data = []

    for page_num, image in enumerate(images):
        # Extract text using OCR
        extracted_text = extract_ocr_text(image)

        # Document classification
        classification_response = requests.post(CLASSIFICATION_API_URL, json={"text": extracted_text})
        classification_label = classification_response.json().get("predicted_label", "Unknown")

        # Table detection using LayoutParser
        layout_model = lp.Detectron2LayoutModel(
            config_path="table-detection_rel_v0.1\config.yaml",
            label_map={0: 'Table'},
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5]
        )
        layout = layout_model.detect(image)
        table_bboxes = [block.coordinates for block in layout if block.type == 'Table']

        # Annotate image with table detection
        annotated_image = image.copy()
        table_csvs = []

        for idx, bbox in enumerate(table_bboxes):
            bbox = [int(coord) for coord in bbox]
            cv2.rectangle(annotated_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(annotated_image, f'Table {idx+1}', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            table_csv = extract_table_data(image, bbox)
            table_csvs.append(table_csv)

        # Signature Detection
        _, img_encoded = cv2.imencode('.jpg', image)
        files = {'file': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')}
        signature_response = requests.post(SIGNATURE_API_URL, files=files)
        signature_data = signature_response.json()

        # Draw signature detections on the annotated image
        for detection in signature_data["detections"]:
            bbox = detection["bounding_box"]
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box for signatures
            cv2.putText(annotated_image, f'Signature {detection["detection_number"]}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Stamp Detection
        stamp_response = requests.post(STAMP_API_URL, files=files)
        stamp_data = stamp_response.json()

        # Draw stamp detections on the annotated image
        for detection in stamp_data["detections"]:
            bbox = detection["bounding_box"]
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box for stamps
            cv2.putText(annotated_image, f'Stamp {detection["detection_number"]}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Upload annotated image (with table, signature, and stamp detections) to S3
        s3_url = upload_to_s3(image_to_bytes(annotated_image), f"annotated_page_{page_num+1}.jpg")

        # Prepare non-table text (all text minus table content)
        non_table_text = extracted_text
        for table_csv in table_csvs:
            non_table_text = non_table_text.replace(table_csv, "")

        # Results
        page_result = {
            "page_number": page_num + 1,
            "s3_url_annotated_image": s3_url,
            "image_classification": classification_label,
            "stamp_count": stamp_data["total_detections"], 
            "signature_count": signature_data["total_detections"],  
            "non_table_text": non_table_text.strip(),
            "table_csvs": table_csvs
        }
        output_data.append(page_result)

    return JSONResponse(content=output_data)
