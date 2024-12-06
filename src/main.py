from fastapi import FastAPI
from routes.generators import router
from routes.langchain import lang_router


app = FastAPI()

app.include_router(router)
app.include_router(lang_router)
