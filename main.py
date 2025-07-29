import os
import re
import tempfile
import requests
import spacy
import pdfplumber
from typing import List
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from pinecone import Pinecone, ServerlessSpec
import openai

# ---------------------------
# Environment setup - secrets
# ---------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = "us-east-1"      # Change if needed
EMBED_MODEL = "text-embedding-3-large"    # Or "text-embedding-ada-002"
LLM_MODEL = "gpt-4o"            # Or "gpt-4-turbo", "gpt-4", etc.

# ---------------------------
# Init clients
# ---------------------------
openai.api_key = OPENAI_API_KEY
nlp = spacy.load("en_core_web_sm")
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "bajajhackrxchsbitshyderabad"
# Use an index per hackathon request, or clear before upserting for statelessness
DIMENSION = 3072   # Embedding size for text-embedding-3-large

# Create Pinecone index if missing
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region=PINECONE_ENV
        )
    )
index = pc.Index(INDEX_NAME)

# ---------------------------
# PDF/Chunking utilities
# ---------------------------
def extract_text_from_pdf_url(url: str) -> str:
    """Download PDF from URL, extract all text."""
    with requests.get(url, stream=True, timeout=15) as r:
        r.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
            temp_pdf = f.name
    try:
        text = ""
        with pdfplumber.open(temp_pdf) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    finally:
        os.remove(temp_pdf)

def split_by_section_markers(text: str):
    # Adjust regex as per your policies for even more robust section splitting
    pattern = r'(?m)^(Section [A-Z0-9]+|[A-Z]\)|\d+\.\d+|\d+\.|[IVXLC]+\.)[ \t-]*(.*)$'
    matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
    sections = []
    if not matches:
        return [("Full Document", text.strip())]
    for i, match in enumerate(matches):
        section_marker = match.group(1).strip()
        section_title = match.group(2).strip()
        start_idx = match.end()
        end_idx = matches[i+1].start() if i+1 < len(matches) else len(text)
        section_text = text[start_idx:end_idx].strip()
        if section_text:
            label = f"{section_marker} {section_title}".strip()
            sections.append((label, section_text))
    return sections

def further_chunk_long_sections(sections, max_words=500):
    final_chunks = []
    for section_label, section_text in sections:
        words = section_text.split()
        if len(words) <= max_words:
            final_chunks.append({
                "section": section_label,
                "text": section_text
            })
        else:
            doc = nlp(section_text)
            sents = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            cur_chunk, cur_len = [], 0
            for sent in sents:
                sent_len = len(sent.split())
                if cur_len + sent_len > max_words and cur_chunk:
                    final_chunks.append({
                        "section": section_label,
                        "text": " ".join(cur_chunk)
                    })
                    cur_chunk, cur_len = [sent], sent_len
                else:
                    cur_chunk.append(sent)
                    cur_len += sent_len
            if cur_chunk:
                final_chunks.append({
                    "section": section_label,
                    "text": " ".join(cur_chunk)
                })
    return final_chunks

# ---------------------------
# Pinecone routines (stateless per request)
# ---------------------------
def get_embedding(text: str) -> List[float]:
    response = openai.embeddings.create(
        input=text,
        model=EMBED_MODEL
    )
    return response.data[0].embedding

def clear_index():
    # Delete all vectors before upsert (for multi-user, consider namespacing by document hash!)
    index.delete(delete_all=True)

def upsert_chunks_to_pinecone(chunks: List[dict]):
    # Use unique ids per run for chunk statelessness
    vectors = [
        (f"chunk_{i}", get_embedding(chunk['text']), {"section": chunk["section"], "text": chunk["text"]})
        for i, chunk in enumerate(chunks)
    ]
    index.upsert(vectors=vectors)

def retrieve_chunks_from_pinecone(query: str, top_k: int = 5):
    q_embed = get_embedding(query)
    res = index.query(vector=q_embed, top_k=top_k, include_metadata=True)
    # Handle v3.x results: `res['matches']`
    return [match["metadata"] for match in res["matches"]]

# ---------------------------
# LLM utilities
# ---------------------------
def build_gpt4_prompt(query: str, retrieved_chunks: List[dict]) -> str:
    formatted_chunks = "\n\n".join(
        [f"Section: {chunk['section']}\n{chunk['text'][:700]}" for chunk in retrieved_chunks]
    )
    prompt = f"""Answer the following question using ONLY the provided policy clauses.
Question: "{query}"

Relevant policy sections/clauses:
{formatted_chunks}

Based only on provided clauses, answer clearly, referencing section labels if possible. If info is missing, say so.

Answer in a single, clear sentence.
"""
    return prompt

def complete_with_gpt4(prompt: str) -> str:
    response = openai.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful insurance policy assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.0
    )
    return response.choices[0].message.content.strip()

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI()

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

@app.post("/hackrx/run")
async def hackrx_run(
    request_data: QueryRequest,
    authorization: str = Header(None)
):
    # -- Bearer Token check
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Bearer Token.")

    # --- Download & index the provided document
    policy_pdf_url = request_data.documents
    try:
        text = extract_text_from_pdf_url(policy_pdf_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error downloading or parsing PDF: {str(e)}")

    # --- Chunk and embed (re-index Pinecone for each request for statelessness)
    sections = split_by_section_markers(text)
    clause_chunks = further_chunk_long_sections(sections, max_words=500)
    clear_index()
    upsert_chunks_to_pinecone(clause_chunks)

    # --- Answer the questions
    answers = []
    for question in request_data.questions:
        chunks = retrieve_chunks_from_pinecone(question, top_k=5)
        prompt = build_gpt4_prompt(question, chunks)
        llm_answer = complete_with_gpt4(prompt)
        answers.append(llm_answer)

    # --- Return in required format
    return {
        "success": True,
        "answers": answers,
        "processing": {
            "num_sections": len(sections),
            "num_chunks": len(clause_chunks),
            "model": LLM_MODEL,
        }
    }
