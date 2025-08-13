# main.py
from typing import TypedDict
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END

# --- 상태 스키마 정의 ---
class AppState(TypedDict, total=False):
    text: str
    summary: str
    keywords: str

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# LLM (로컬)
llm = ChatOllama(model="llama3", temperature=0.2)

summary_prompt = PromptTemplate(
    input_variables=["content"],
    template="다음 텍스트를 한 문장으로 요약해 주세요:\n{content}"
)
keywords_prompt = PromptTemplate(
    input_variables=["summary"],
    template="다음 요약문에서 핵심 키워드 3~5개를 쉼표로 구분해서 추출해 주세요:\n{summary}"
)

# --- LangGraph 노드 정의 ---
def summarize_node(state: AppState):
    prompt = summary_prompt.format(content=state.get("text", ""))
    response = llm.invoke(prompt)
    # Partial state 반환: summary만 업데이트
    return {"summary": response.content}

def keywords_node(state: AppState):
    prompt = keywords_prompt.format(summary=state.get("summary", ""))
    response = llm.invoke(prompt)
    return {"keywords": response.content}

# --- StateGraph 초기화: state_schema를 반드시 전달 ---
graph = StateGraph(state_schema=AppState)
graph.add_node("summarize", summarize_node)
graph.add_node("keywords", keywords_node)
graph.set_entry_point("summarize")
graph.add_edge("summarize", "keywords")
graph.add_edge("keywords", END)

compiled_graph = graph.compile()

# --- FastAPI 라우트 ---
@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "summary": None,
        "keywords": None
    })

@app.post("/", response_class=HTMLResponse)
async def submit_form(request: Request, text: str = Form(...)):
    # 초기 상태로 text 전달
    result = compiled_graph.invoke({"text": text})
    # result는 상태 전체 (dict)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "summary": result.get("summary"),
        "keywords": result.get("keywords")
    })
