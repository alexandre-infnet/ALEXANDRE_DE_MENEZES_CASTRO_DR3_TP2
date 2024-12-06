from fastapi import APIRouter, HTTPException
from transformers import pipeline

from models.InputData import InputData
from models.TranslationRequest import TranslationRequest


router = APIRouter()


@router.post("/generate-text/")
async def generate_text(data: InputData):
    text_generator = pipeline("text-generation", model="gpt2")

    try:
        generated_texts = text_generator(
            data.prompt,
            max_length=data.max_length,
            num_return_sequences=data.num_return_sequences
        )

        return {"generated_texts": [result["generated_text"] for result in generated_texts]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/translate/")
async def translate_text(request: TranslationRequest):
    translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")

    try:
        translated = translator(request.text, max_length=500)
        translated_text = translated[0]["translation_text"]

        return {"translated_text": translated_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
