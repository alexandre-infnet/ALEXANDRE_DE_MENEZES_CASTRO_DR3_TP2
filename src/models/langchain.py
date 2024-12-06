from fastapi import APIRouter

from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import FakeListLLM, HuggingFaceHub

from models.UserInput import UserInput
from models.TranslationRequest import TranslationRequest


lang_router = APIRouter()

responses = [
    "Olá! Como posso ajudar você?",
    "Eu sou um chatbot simples criado com Fake LLM.",
    "Eu simulo respostas de um LLM para testes rápidos.",
    "Desculpe, não entendi sua pergunta."
]

fake_llm = FakeListLLM(responses=responses)


@lang_router.post("/chat")
async def chat(user_input: UserInput):
    response = fake_llm.invoke(user_input.prompt)

    return {"response": response}


@lang_router.post("/translate_lang/")
async def translate(text_input: TranslationRequest):
    original_text = text_input.text

    template = ChatPromptTemplate([
        ("system", "You are an English to French translator. Reject any other language."),
        ("user", "Translate this: {text}")
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", api_key="")
    response = llm.invoke(template.format_messages(text=original_text))

    return {"response": response.content}


@lang_router.post("/translate-german", summary="Translate English text to German")
async def translate_german(request: TranslationRequest):
    llm = HuggingFaceHub(
        repo_id='Helsinki-NLP/opus-mt-en-de',
        huggingfacehub_api_token="",
        model_kwargs={
            "max_length": 256,
            "num_beams": 4
        }
    )

    output = llm.invoke(request.text)

    return {"german_translation": output}
