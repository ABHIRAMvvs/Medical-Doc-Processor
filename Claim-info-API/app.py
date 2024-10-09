from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io
import numpy as np
from paddleocr import PaddleOCR
import requests
import json
import traceback
import logging
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize the PaddleOCR model
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def extract_text_from_image(image_bytes):
    """
    Extract text from an image using PaddleOCR.
    
    :param image_bytes: bytes, image file content
    :return: str, extracted text from the image
    """
    try:
        # Convert bytes to numpy array
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)
        
        # Use PaddleOCR to extract text from the image
        result = ocr.ocr(image_np, cls=True)
        
        # Concatenate all the recognized text lines into a single string
        extracted_text = "\n".join([line[1][0] for line in result[0]])
        
        return extracted_text
    except Exception as e:
        logger.error(f"Error in extract_text_from_image: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def determine_document_type(extracted_text):
    """
    Determine if the document is a discharge summary, an invoice, or a cheque based on keywords in the extracted text.
    
    :param extracted_text: str, text extracted from the document
    :return: str, either 'discharge_summary', 'invoice', or 'cheque'
    """
    if "diagnosis" in extracted_text.lower() or "treatment" in extracted_text.lower():
        return "discharge_summary"
    elif "amount" in extracted_text.lower() or "price" in extracted_text.lower() or "unit" in extracted_text.lower():
        return "invoice"
    elif "payee" in extracted_text.lower() or "account" in extracted_text.lower() or "ifsc" in extracted_text.lower() or "branch" in extracted_text.lower():
        return "cheque"
    return None

def process_medical_text(extracted_text):
    """
    Process the extracted text by sending it to a model and returning structured JSON based on document type.
    Clears Llama context state for each call.
        
    :param extracted_text: str, extracted text from the medical report, invoice, or cheque
    :return: dict, processed information in JSON format
    """
    doc_type = determine_document_type(extracted_text)

    if doc_type == "discharge_summary":
        prompt = f"""
        You are a medical expert tasked with extracting key information from medical reports. Only give the JSON output.
        Extract the following information from the medical report and return it in JSON format, strictly adhering to this structure:
        {{
            "diagnosis": [
                {{
                    "name": "DIAGNOSIS NAME IN UPPERCASE",
                    "icd": "ICD CODE"
                }},
                ...
            ],
            "symptoms": [
                "symptom description in lowercase"
            ],
            "treatment_taken": "Treatment description"
        }}
        
        Ensure that:
        1. ICD codes are included if mentioned, otherwise omit the "icd" field.
        2. Diagnosis contains the diseases and infections that have been deduced for the patient and check the complete context for them.
        3. If the words for Diagnosis are in short form, use only the long form of the words.
        3. Symptoms are multiple so check for the complete context for them. 
        4. There can be multiple diagnoses and symptoms, but only one treatment_taken.
        5. If you can't find any value for the  keys, then leave them as empty lists.
        
        Medical Report: {extracted_text}
        """
    elif doc_type == "invoice":
        prompt = f"""
        You are tasked with extracting key information from medical invoices. Only give the JSON output.
        Extract the following information from each row and return it in JSON format:
        - claimedAmount: Claimed amount for each item
        - date: Date of the entry
        - particular: Description of the item
        - price: Price per unit
        - isPayable: Is the item payable (true/false)
        - unit: Number of units
        
        Invoice: {extracted_text}
        """
    elif doc_type == "cheque":
        prompt = f"""
        You are tasked with extracting key financial details from a cheque. Only give the JSON output.
        Extract the following information and return it in JSON format:
        - payee_name: The name of the payee
        - bank_name: The name of the bank
        - account_number: The account number
        - ifsc_code: The IFSC code of the bank
        - branch: The branch name
        - bank_address: The address of the bank
        - state: The state where the bank is located
        - city: The city where the bank is located
        - pin: The PIN code of the bank's location
        
        Cheque: {extracted_text}
        """
    else:
        return None
    
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "llama3.1",
        "prompt": prompt,
        "options": {
            "reset_context": True  # This option clears the context for each new request
        }
    }
    
    try:
        response = requests.post(url, json=data, stream=True)
        response.raise_for_status()
        
        complete_response = ""
        for line in response.iter_lines(decode_unicode=True):
            if line:
                json_line = json.loads(line)
                complete_response += json_line.get('response', '')

        logger.info(f"Complete response from Llama: {complete_response}")

        # Remove code block markers if present
        complete_response = complete_response.replace('```json', '').replace('```', '').strip()

        json_response = json.loads(complete_response)
        return json_response
    except requests.RequestException as e:
        logger.error(f"Error communicating with Llama server: {str(e)}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from Llama response: {str(e)}")
        logger.error(f"Raw response: {complete_response}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in process_medical_text: {str(e)}")
        logger.error(traceback.format_exc())
        return None

@app.post("/get-claim-info/")
async def get_claim_info(file: UploadFile = File(...)):
    logger.info(f"Received file: {file.filename}")
    try:
        contents = await file.read()
        logger.info(f"File size: {len(contents)} bytes")
        
        extracted_text = extract_text_from_image(contents)
        if not extracted_text:
            logger.warning("No text extracted from the image.")
            return JSONResponse(status_code=400, content={"message": "No text extracted from the image."})
        
        logger.info(f"Extracted text: {extracted_text[:100]}...")
        information = process_medical_text(extracted_text)
        if not information:
            logger.warning("Failed to process the extracted text.")
            return JSONResponse(status_code=400, content={"message": "Failed to process the extracted text."})
        
        logger.info(f"Processed information: {information}")
        return JSONResponse(status_code=200, content=information)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}", "traceback": traceback.format_exc()})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8083)