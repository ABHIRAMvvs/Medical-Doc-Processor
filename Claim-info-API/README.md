# Medical Document Processing API

This API provides functionality to process medical documents, including discharge summaries, invoices, and cheques. It uses optical character recognition (OCR) to extract text from images and then processes the extracted text to provide structured information.

## Setup

### Prerequisites

- Python 3.11.7
- [OLLAMA](https://ollama.com/)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- FastAPI
- Uvicorn

### Installation

1. Clone the repository:
   ```
   git clone https://saiabhiram16vadrevus@bitbucket.org/clarion-primevision/vertexview-e2e-backend-pipeline.git
   cd VertexView-E2E-Backend-Pipeline/Claim-info-API
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install OLLAMA:
   Follow the instructions on the [OLLAMA Download page](https://ollama.com/download) to install OLLAMA on your system.

4. Download the llama3.1 model from the Ollama library:
   ```
   ollama pull llama3.1
   ```

### Running the API

1. Start the OLLAMA server:
   ```
   ollama serve
   ```

2. In a new terminal, start the FastAPI server:
   ```
   uvicorn app:app --host 0.0.0.0 --port 8083
   ```

The API will be available at `http://IP-address:8083`.

## API Documentation

### Endpoint: `/get-claim-info/`

- **Method**: POST
- **Description**: Processes an uploaded image of a medical document (discharge summary, invoice, or cheque) and returns structured information for claim processing.
- **Request**:
  - Content-Type: `multipart/form-data`
  - Body: 
    - `file`: The image file to be processed (supported formats: JPEG, PNG)
- **Response**:
  - Content-Type: `application/json`
  - Body: JSON object containing the extracted and processed information, structure depends on the document type

#### Response Structures

1. Discharge Summary:
   ```json
   {
     "diagnosis": [
       {
         "name": "string",
         "icd_code": "string"
       }
     ],
     "symptoms": [
       "string"
     ],
     "treatment_taken": "string"
   }
   ```

2. Invoice:
   ```json
   {
     "items": [
       {
         "claimedAmount": "number",
         "date": "string",
         "particular": "string",
         "price": "number",
         "isPayable": "boolean",
         "unit": "number"
       }
     ]
   }
   ```

3. Cheque:
   ```json
   {
     "payee_name": "string",
     "bank_name": "string",
     "account_number": "string",
     "ifsc_code": "string",
     "branch": "string",
     "bank_address": "string",
     "state": "string",
     "city": "string",
     "pin": "string"
   }
   ```

## Usage

To use the API, send a POST request to the `/get-claim-info/` endpoint with the image file in the request body. You can use tools like cURL, Postman, or any programming language with HTTP client capabilities.

Example using cURL:

```bash
curl -X POST "http://IP-address:8000/get-claim-info/" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@path/to/your/image.jpg"
```

## Error Handling

The API includes error handling for various scenarios:

- If no text is extracted from the image, a 400 status code is returned with an appropriate message.
- If processing the extracted text fails, a 400 status code is returned with an error message.
- For any unexpected errors, a 500 status code is returned with the error message and traceback.