# api/http_api.py
import os
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from ..memory.long_term import LongTermMemory
from ..memory.short_term import ShortTermMemory
from ..rag.index import RAGIndex
from ..rag.loaders.pdf_loader import load_pdf_as_text
from ..rag.loaders.swagger_loader import fetch_swagger_json, summarise_swagger
from ..agent.core import TestWeaverAgent

app = FastAPI(
    title="TestWeaver Agent API",
    description="Agent for code-aware test case generation with RAG + MCP Git",
    version="0.1.0"
)

lt_memory = LongTermMemory()
st_memory = ShortTermMemory()
rag_index = RAGIndex(lt_memory)

SVC_REPO = os.getenv("GIT_REPO_SVC_ACCOUNTING", "your-org/svc-accounting")

class ChatRequest(BaseModel):
    session_id: str
    message: str
    query_for_rag: str | None = None

class GenerateTestsRequest(BaseModel):
    session_id: str
    service_path: str
    extra_instructions: str | None = None

@app.post("/chat")
def chat(req: ChatRequest):
    agent = TestWeaverAgent(req.session_id, rag_index, st_memory, SVC_REPO)
    answer = agent.chat(req.message, query_for_rag=req.query_for_rag)
    return {"reply": answer}

@app.post("/generate-tests")
def generate_tests(req: GenerateTestsRequest):
    agent = TestWeaverAgent(req.session_id, rag_index, st_memory, SVC_REPO)
    code = agent.generate_tests_for_file(req.service_path, extra_instructions=req.extra_instructions or "")
    return {"test_code": code}

@app.post("/ingest/pdf")
async def ingest_pdf(session_id: str = Form(...), file: UploadFile = File(...)):
    content = await file.read()
    temp_path = f"./data/docs/{file.filename}"
    os.makedirs("./data/docs", exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(content)

    text = load_pdf_as_text(temp_path)
    doc_id = f"pdf:{file.filename}"
    rag_index.ingest_text(doc_id, text, meta={"type": "pdf", "session_id": session_id})
    return {"status": "ok", "doc_id": doc_id}

@app.post("/ingest/swagger")
def ingest_swagger(url: str):
    openapi = fetch_swagger_json(url)
    text = summarise_swagger(openapi)
    doc_id = f"swagger:{url}"
    rag_index.ingest_text(doc_id, text, meta={"type": "swagger", "url": url})
    return {"status": "ok", "doc_id": doc_id}
