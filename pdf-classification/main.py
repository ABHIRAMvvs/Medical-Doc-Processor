import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
from pydantic import BaseModel

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

labels = [
    'account_details', 'cancelled_cheque', 'cashless_authorization_letter', 'claim_cover_letter', 
    'claim_form', 'diagnostic_report', 'discharge_summary', 'history', 'invoice', 'kyc', 
    'lab_report', 'member_card', 'miscellaneous', 'pharmacy_bill', 'prescription', 
    'progress_notes', 'signed_claim_form'
]
label2id = {label: id for id, label in enumerate(labels)}
id2label = {id: label for label, id in label2id.items()}

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

model = RobertaForSequenceClassification.from_pretrained(
    'roberta-base',
    num_labels=len(labels)
)

model.load_state_dict(torch.load("./pdf-classification-model/pdf-classification_rel_v0.1.pth", map_location=device))
model = model.to(device)
model.eval()

class DocumentInput(BaseModel):
    text: str

def predict_label(text):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        conf_score, preds = torch.max(probs, dim=1)

        predicted_label = id2label[preds.item()]
        confidence = conf_score.item()

        return predicted_label, confidence*100

@app.post("/classify_document_type")
async def classify_document_type(document: DocumentInput):
    try:
        text = document.text
        if not text:
            raise HTTPException(status_code=400, detail="No text provided.")

        predicted_label, confidence = predict_label(text)
        return JSONResponse({"predicted_label": predicted_label, "confidence": confidence})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
