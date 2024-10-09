# Medical Doc Processor

This is an advanced PDF processing API that leverages OCR, table detection, document classification, signature detection, and stamp detection to extract and analyze content from PDF documents.

## Features

- PDF to image conversion
- OCR text extraction
- Document classification
- Table detection and structure analysis
- Signature detection
- Stamp detection
- Image annotation with detected tables, signatures, stamps, and other structural elements
- Table data extraction to CSV format
- Processed image storage in Amazon S3

## Workflow Overview

The API performs the following steps for each uploaded PDF file:

1. **PDF to Image Conversion**
   - The PDF file is converted into a series of images using `PyMuPDF`.
   - Each page in the PDF is converted to an image for further processing.

2. **Text Extraction using OCR**
   - The images are processed using `PaddleOCR` to extract textual content.

3. **Document Classification**
   - The extracted text is sent to an external document classification API.
   - The API responds with a label predicting the document type (e.g., "Invoice", "Discharge Summary").

4. **Table Detection and Extraction**
   - Using custom `layoutlmv3` model and `layoutparser` library, tables are detected in the images.
   - The table content is extracted and converted into CSV format for easy analysis.

5. **Signature Detection**
   - Signature detection is performed by sending the images to an external signature detection API.
   - The detected signature locations are drawn on the image.

6. **Stamp Detection**
   - Stamp detection is performed by sending the images to an external stamp detection API.
   - The detected stamp locations are drawn on the image.

7. **Image Annotation**
   - Detected tables, signatures, and stamps are annotated on the images.
   - Annotated images are uploaded to an AWS S3 bucket, and URLs are generated.

8. **Results Compilation**
   - A structured JSON response is generated, containing:
     - URLs to the annotated images
     - Document classification results
     - Signature and stamp detection counts
     - Non-table text content
     - Extracted table content as CSV.

## Requirements

1. This project was built using Python version 3.11.7. Before implementing the project, install compatible PyTorch and Detectron2 libraries using the following commands:

    ```bash
    pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 torchaudio==2.4.1+cu118 --index-url https://download.pytorch.org/whl/cu118
    ```

    ```bash
    python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
    ```

2. Install the required libraries using the **requirements.txt** file and the following command:
   
    ```bash
    pip install -r requirements.txt
    ```

## Setup Instructions
The necessary resources for table detection, structural recognition and pdf classification are stored in an S3 bucket. Download the required zip files from the following path of the S3 bucket:

 - Table Detection Model: `/abhiram-s3/vertex-view-resources/table-detection.zip`
 - Structural Transformer Model: `/abhiram-s3/vertex-view-resources/structural-transformer.zip`
 - Pdf classification model: `/abhiram-s3/vertex-view-resources/pdf-classification_rel_v0.1.pth`

**Steps to Setup:** 
- Download the zip files from the above links.
- Unzip the files and place them in the `table-detection_rel_v0.1/`, `structural-transformer_rel_v0.1/` and `pdf-classification/pdf-classification-model/pdf-classification_rel_v0.1.pth` respectively. 

## Usage

1. Start the FastAPI server:
   
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```
    The API will be available at `http://162.19.116.128:8000/`. Use the `/process_pdf/` endpoint to submit PDF files for processing.

2. Start the FastAPI server for pdf-classification API. The codebase for pdf classification is available in a sub-directory called `pdf-classification`:

    ```bash
    cd pdf-classification
    uvicorn main:app --relaod --port 8080
    ```
    The API will be available at `http://127.0.0.1:8080/`. Use the `/classify_document_type` endpoint to submit PDF files for processing.

3. Start the FastAPI server for signature-detection API. The codebase for signature detection is available in a sub-directory called `signature-detection`:

    ```bash
    cd signature-detection
    uvicorn app:app --reload --port 8081
    ```
    The API will be available at `http://127.0.0.1:8081/`. Use the `/detect_signtures/` endpoint to submit PDF files for processing.

4. Start the FastAPI server for stamp-detection API. The codebase for stamp detection is available in a sub-directory called `stamp-detection`:

    ```bash
    cd stamp-detection
    uvicorn app:app --reload --port 8082
    ```
    The API will be available at `http://127.0.0.1:8082/`. Use the `/detect_stamps` endpoint to submit PDF files for processing.

4. You can use the provided test.py script to test the API. The script contains pre-configured requests and validation steps. To run the test:

    ```bash
    python test.py
    ```
    The test document required for API testing can be downloaded from the following path in the S3 bucket:
    - Test Document: /abhiram-s3/vertex-view-resources/test.pdf
    
    Make sure to place the downloaded test.pdf in the required directory or update the script with the correct path.

## API Documentation

### Endpoint: Process PDF

Processes a PDF document, performing OCR, table detection, document classification, signature detection, and stamp detection.

- **URL:** `/process_pdf/`
- **Method:** POST
- **Content-Type:** multipart/form-data

### Request

| Parameter | Type | Description |
|-----------|------|-------------|
| pdf | File | The PDF file to be processed |

### Response

Returns a JSON array with information for each page in the PDF:

```json
[
  {
    "page_number": 1,
    "s3_url_annotated_image": "https://s3.sgp.io.cloud.ovh.net/abhiram-s3/vertex-view-resources/annotated-images/annotated_page_1.jpg",
    "image_classification": "Invoice",
    "stamp_count": 2,
    "signature_count": 1,
    "non_table_text": "This is the extracted text excluding table content.",
    "table_csvs": [
      "Header1,Header2,Header3\nRow1Col1,Row1Col2,Row1Col3\nRow2Col1,Row2Col2,Row2Col3"
    ]
  },
  // ... (one object per page)
]
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| page_number | Integer | The page number in the PDF |
| s3_url_annotated_image | String | URL of the annotated image stored in S3 |
| image_classification | String | Classification label for the document |
| stamp_count | Integer | Number of stamps detected on the page |
| signature_count | Integer | Number of signatures detected on the page |
| non_table_text | String | Extracted text content, excluding table data |
| table_csvs | Array of Strings | CSV representation of detected tables |

### Error Responses

- **400 Bad Request:** If the PDF file is missing or invalid
- **500 Internal Server Error:** If there's an error processing the PDF
