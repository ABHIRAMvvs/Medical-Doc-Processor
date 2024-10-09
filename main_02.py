from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import fitz
import cv2
import layoutparser as lp
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import requests
import logging
import boto3
from transformers import TableTransformerForObjectDetection, DetrImageProcessor
import torch
from torchvision import transforms
from tqdm.auto import tqdm

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

S3_ACCESS_KEY = 'd812703636ea4510a0c7651623e5a386'
S3_SECRET_KEY = '2eab9ce147cb4671b00fe9ab37ca982e'
S3_ENDPOINT = 'https://s3.sgp.io.cloud.ovh.net'
S3_BUCKET_NAME = 'abhiram-s3'
REGION = 'sgp'

ocr = PaddleOCR(use_angle_cls=True, lang='en')
device = "cuda" if torch.cuda.is_available() else "cpu"
structure_model = TableTransformerForObjectDetection.from_pretrained("structural-transformer_rel_v0.1")
structure_model.to(device)

class MaxResize(object):

    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale*width)), int(round(scale*height))))
        return resized_image

structure_transform = transforms.Compose([
    MaxResize(1000),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

layout_model = lp.Detectron2LayoutModel(
    config_path="table-detection_rel_v0.1/config.yaml",
    label_map={0: 'Table'},
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5]
)

s3_client = boto3.client(
    's3',
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    endpoint_url=S3_ENDPOINT,
    region_name=REGION
)

CLASSIFICATION_API_URL = "http://127.0.0.1:8080/classify_document_type"
SIGNATURE_API_URL = "http://127.0.0.1:8081/detect_signatures/"
STAMP_API_URL = "http://127.0.0.1:8082/detect_stamps/"

def upload_to_s3(file_content: bytes, file_name: str) -> str:
    s3_key = f"vertex-view-resources/annotated-images/{file_name}"
    s3_client.put_object(Body=file_content, Bucket=S3_BUCKET_NAME, Key=s3_key)
    s3_url = f"{S3_ENDPOINT}/{S3_BUCKET_NAME}/{s3_key}"
    return s3_url

def image_to_bytes(image: np.ndarray) -> bytes:
    _, buffer = cv2.imencode('.jpg', image)
    return buffer.tobytes()

def outputs_to_objects(outputs, img_size, id2label):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if class_label != 'no object':
            objects.append({'label': class_label, 'score': float(score), 'bbox': [float(elem) for elem in bbox]})

    return objects

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def get_cell_coordinates(table_data):
    rows = [entry for entry in table_data if entry['label'] == 'table row']
    columns = [entry for entry in table_data if entry['label'] == 'table column']
    rows.sort(key=lambda x: x['bbox'][1])
    columns.sort(key=lambda x: x['bbox'][0])
    def find_cell_coordinates(row, column):
        return [max(column['bbox'][0], row['bbox'][0]), 
                max(column['bbox'][1], row['bbox'][1]), 
                min(column['bbox'][2], row['bbox'][2]), 
                min(column['bbox'][3], row['bbox'][3])]
    cell_coordinates = []
    for i, row in enumerate(rows):
        row_cells = []
        for j, column in enumerate(columns):
            cell_bbox = find_cell_coordinates(row, column)
            if cell_bbox[2] - cell_bbox[0] > 1 and cell_bbox[3] - cell_bbox[1] > 1:
                row_cells.append({
                    'row_index': i,
                    'col_index': j,
                    'cell': cell_bbox
                })
        cell_coordinates.append({
            'row': row['bbox'],
            'cells': row_cells,
            'cell_count': len(row_cells)
        })
    return cell_coordinates

def apply_ocr(cell_coordinates, cropped_table):
    data = dict()
    for idx, row in enumerate(tqdm(cell_coordinates)):
        row_text = []
        for cell in row["cells"]:
            cell_image = np.array(cropped_table.crop(cell["cell"]))
            result = ocr.ocr(cell_image, cls=True)
            text = " ".join([line[1][0] for line in result[0]]) if result[0] else ""
            row_text.append(text)
        data[idx] = row_text
    return data

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
        extracted_text = ocr.ocr(image)
        classification_response = requests.post(CLASSIFICATION_API_URL, json={"text": extracted_text})
        classification_label = classification_response.json().get("predicted_label", "Unknown")
        layout = layout_model.detect(image)
        table_bboxes = [block.coordinates for block in layout if block.type == 'Table']
        annotated_image = image.copy()
        table_csvs = []
        for idx, bbox in enumerate(table_bboxes):
            bbox = [int(coord) for coord in bbox]
            cropped_image = Image.fromarray(image[bbox[1]:bbox[3], bbox[0]:bbox[2]])
            pixel_values = structure_transform(cropped_image).unsqueeze(0).to(device)
            with torch.no_grad():
                structure_outputs = structure_model(pixel_values)
            structure_id2label = structure_model.config.id2label
            structure_id2label[len(structure_id2label)] = "no object"
            table_data = outputs_to_objects(structure_outputs, cropped_image.size, structure_id2label)
            cell_coordinates = get_cell_coordinates(table_data)
            table_data_extracted = apply_ocr(cell_coordinates, cropped_image)
            csv_content = "\n".join([",".join(row) for row in table_data_extracted.values()])
            table_csvs.append(csv_content)
            cv2.rectangle(annotated_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(annotated_image, f'Table {idx+1}', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        _, img_encoded = cv2.imencode('.jpg', image)
        files = {'file': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')}
        signature_response = requests.post(SIGNATURE_API_URL, files=files)
        signature_data = signature_response.json()
        for detection in signature_data["detections"]:
            bbox = detection["bounding_box"]
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(annotated_image, f'Signature {detection["detection_number"]}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        stamp_response = requests.post(STAMP_API_URL, files=files)
        stamp_data = stamp_response.json()
        for detection in stamp_data["detections"]:
            bbox = detection["bounding_box"]
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  
            cv2.putText(annotated_image, f'Stamp {detection["detection_number"]}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        s3_url = upload_to_s3(image_to_bytes(annotated_image), f"annotated_page_{page_num+1}.jpg")

        if isinstance(extracted_text, list):
            non_table_text = " ".join(
                str(item[1][0]) for sublist in extracted_text for item in sublist if isinstance(item, list) and len(item) > 1
            )
        else:
            non_table_text = str(extracted_text)  
        for table_csv in table_csvs:
            non_table_text = non_table_text.replace(table_csv, "")
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