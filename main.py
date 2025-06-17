from fastapi import FastAPI, Request, Form, UploadFile, File, requests
import requests
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, JSONResponse
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.messages import HumanMessage
from deepgram import Deepgram
from fastapi.staticfiles import StaticFiles
import os, json, fitz, aiofiles

from moke import create_retriever

# --- Setup ---
load_dotenv()
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(SessionMiddleware, secret_key="your_secret_key")
templates = Jinja2Templates(directory="templates")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

dg_client = Deepgram(DEEPGRAM_API_KEY)

llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-70b-8192", streaming=True)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
retriever = create_retriever()
CHUNK_SIZE = 200
CHUNK_OVERLAP = 20
PDF_VECTOR_DIR = "pdf_vector_db"

# --- Utilities ---
def load_history():
    try:
        with open("chat_history.json", "r") as f:
            return json.load(f)
    except:
        return []

def save_chat(question, answer):
    history = load_history()
    history.append({"question": question, "answer": answer})
    with open("chat_history.json", "w") as f:
        json.dump(history, f, indent=2)

# --- Tools ---
fashion_tool = Tool(name="Fashion Tool", func=lambda q: RetrievalQA.from_chain_type(llm=llm, retriever=retriever).run(q), description="Fashion questions")
general_tool = Tool(name="Global_QA", func=lambda q: llm.invoke(q).content, description="General knowledge")
program_tool = Tool(name="Program Tool", func=lambda q: llm.invoke(f"You're a coding expert. Please write the code for: {q}").content, description="Programming")

fashion_agent = initialize_agent([fashion_tool], llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
general_agent = initialize_agent([general_tool], llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
program_agent = initialize_agent([program_tool], llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# --- Dummy user DB ---
USERS = {"admin": "admin123"}

# --- Routes ---
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": None, "logged_in": False})

@app.post("/login", response_class=HTMLResponse)
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if USERS.get(username) == password:
        request.session["user"] = username
        return templates.TemplateResponse("index.html", {"request": request, "history": load_history(), "logged_in": True})
    else:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid username or password", "logged_in": False})

@app.get("/logout", response_class=HTMLResponse)
async def logout(request: Request):
    request.session.clear()
    return templates.TemplateResponse("login.html", {"request": request, "error": "You have been logged out.", "logged_in": False})

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    if not request.session.get("user"):
        return RedirectResponse("/login")
    return templates.TemplateResponse("index.html", {"request": request, "history": load_history(), "logged_in": True})

@app.post("/upload_pdf", response_class=HTMLResponse)
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    if not request.session.get("user"):
        return RedirectResponse("/login")

    try:
        contents = await file.read()
        with open("uploaded.pdf", "wb") as f:
            f.write(contents)

        text = ""
        with fitz.open("uploaded.pdf") as doc:
            for page in doc:
                text += page.get_text()

        splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        docs = splitter.create_documents([text])

        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(PDF_VECTOR_DIR)

        request.session["pdf_uploaded"] = True
        request.session["just_uploaded_pdf"] = True

        return templates.TemplateResponse("index.html", {"request": request, "answer": "✅ PDF uploaded and processed successfully!", "history": load_history(), "logged_in": True})

    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "answer": f"❌ Error uploading PDF: {e}", "history": load_history(), "logged_in": True})

@app.post("/ask", response_class=HTMLResponse)
async def ask(request: Request, question: str = Form(...)):
    if not request.session.get("user"):
        return RedirectResponse("/login")

    try:
        q_lower = question.lower()
        fashion_keywords = ["clothing", "dress", "fabric", "style", "runway", "fashion", "wear", "trends", "model"]
        programming_keywords = ["code", "python", "function", "loop", "bug", "compile", "variable", "error", "logic"]

        history = load_history()
        conversation = "\n".join([f"User: {h['question']}\nBot: {h['answer']}" for h in history])

        if request.session.get("just_uploaded_pdf"):
            vectorstore = FAISS.load_local(PDF_VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
            pdf_qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
            answer = pdf_qa.run(question)
            note = "📄 Answer from uploaded PDF:"
            request.session["just_uploaded_pdf"] = False

        elif any(word in q_lower for word in fashion_keywords):
            answer = fashion_tool.func(f"{conversation}\n\n{question}")
            note = "👗 Fashion Answer:"

        elif any(word in q_lower for word in programming_keywords):
            answer = program_tool.func(f"{conversation}\n\n{question}")
            note = "💻 Programming Answer:"

        else:
            answer = general_tool.func(f"{conversation}\n\n{question}")
            note = "🌐 General Knowledge Answer:"

        full_answer = f"{note}\n\n{answer}"
        save_chat(question, answer)

        return templates.TemplateResponse("index.html", {"request": request, "question": question, "answer": full_answer, "source_docs": [], "history": load_history(), "logged_in": True})

    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "question": question, "answer": f"❌ Error: {e}", "source_docs": [], "history": load_history(), "logged_in": True})

@app.post("/clear", response_class=HTMLResponse)
async def clear(request: Request):
    if not request.session.get("user"):
        return RedirectResponse("/login")
    with open("chat_history.json", "w") as f:
        json.dump([], f)
    return templates.TemplateResponse("index.html", {"request": request, "history": [], "logged_in": True})

@app.get("/stream")
async def stream_response(query: str):
    async def generate():
        async for chunk in llm.astream(HumanMessage(content=query)):
            yield chunk.content
    return StreamingResponse(generate(), media_type="text/plain")

@app.post("/voice_chat")
async def voice_chat(request: Request, file: UploadFile = File(...)):
    audio_data = await file.read()

    # Step 1: Transcribe using Deepgram
    response = requests.post(
        "https://api.deepgram.com/v1/listen",
        headers={
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": file.content_type  
        },
        data=audio_data
    )

    if response.status_code != 200:
        return JSONResponse({"error": "Transcription failed", "details": response.text}, status_code=500)

    result = response.json()
    transcription = result["results"]["channels"][0]["alternatives"][0]["transcript"]

    q_lower = transcription.lower()
    fashion_keywords = ["clothing", "dress", "fabric", "style", "runway", "fashion", "wear", "trends", "model"]
    programming_keywords = ["code", "python", "function", "loop", "bug", "compile", "variable", "error", "logic"]

    if request.session.get("just_uploaded_pdf"):
        vectorstore = FAISS.load_local(PDF_VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
        pdf_qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
        answer = pdf_qa.run(transcription)
        request.session["just_uploaded_pdf"] = False

    elif any(word in q_lower for word in fashion_keywords):
        answer = fashion_tool.func(transcription)

    elif any(word in q_lower for word in programming_keywords):
        answer = program_tool.func(transcription)

    else:
        answer = general_tool.func(transcription)

    save_chat(transcription, answer)

    return JSONResponse({"question": transcription, "answer": answer})
