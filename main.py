from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate

app = FastAPI()

# Static / Template 설정
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# LLM (로컬 Llama3 모델)
llm = ChatOllama(model="llama3", temperature=0.2)

# 프롬프트 템플릿
prompt_template = PromptTemplate(
    input_variables=["content"],
    template="다음 텍스트를 한 문장으로 요약해 주세요:\n{content}"
)

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "summary": None})

@app.post("/", response_class=HTMLResponse)
async def submit_form(request: Request, text: str = Form(...)):
    prompt = prompt_template.format(content=text)
    response = llm.invoke(prompt)
    return templates.TemplateResponse("index.html", {"request": request, "summary": response.content})
