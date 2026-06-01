import os
import faiss
import pdfplumber
import numpy as np

from dotenv import load_dotenv
from openai import OpenAI

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =========================================================
# LOAD OPENAI API KEY
# =========================================================

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

client = OpenAI(api_key=API_KEY)

# =========================================================
# PDF PATH
# =========================================================

PDF_PATH = r"/Users/rahilmehta/Desktop/untitled folder/data.pdf/data.pdf"

# =========================================================
# EXTRACT TEXT FROM PDF
# =========================================================

print("\nOpening PDF...\n")

full_text = ""

with pdfplumber.open(PDF_PATH) as pdf:
    for page_num, page in enumerate(pdf.pages):
        print(f"Processing Page {page_num + 1}")
        text = page.extract_text()
        if text:
            full_text += text + "\n"

print("\nTEXT EXTRACTION COMPLETED\n")
print(f"TOTAL EXTRACTED TEXT LENGTH: {len(full_text)}")

# =========================================================
# SAVE TEXT
# =========================================================

os.makedirs("data", exist_ok=True)

with open("data/extracted_text.txt", "w", encoding="utf-8") as f:
    f.write(full_text)

print("Extracted text saved")

# =========================================================
# CREATE CHUNKS
# =========================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_text(full_text)

chunks = [c for c in chunks if len(c.strip()) > 100]

print(f"Total chunks created: {len(chunks)}")

# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

print("\nLoading embedding model...\n")

model = SentenceTransformer("BAAI/bge-small-en")

# =========================================================
# CREATE EMBEDDINGS (FIXED)
# =========================================================

print("\nGenerating embeddings...\n")

embeddings = model.encode(
    chunks,
    show_progress_bar=True,
    normalize_embeddings=True
)

embeddings = np.array(embeddings).astype("float32")

print("Embeddings created successfully")

# =========================================================
# CREATE FAISS INDEX
# =========================================================

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # cosine-like search

index.add(embeddings)

print("FAISS database ready")

# =========================================================
# CHAT LOOP
# =========================================================

print("\n===================================")
print("AI EQUITY RESEARCH ASSISTANT READY")
print("Type 'exit' to stop")
print("===================================\n")

chat_history = []

while True:

    query = input("Question: ")

    if query.lower() == "exit":
        print("Exiting...")
        break

    # =====================================================
    # QUERY EMBEDDING (FIXED)
    # =====================================================

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.array(query_embedding).astype("float32")

    # =====================================================
    # SEARCH FAISS
    # =====================================================

    D, I = index.search(query_embedding, k=12)

    retrieved_chunks = []

    for idx in I[0]:
        if idx == -1:
            continue
        if idx < len(chunks):
            retrieved_chunks.append(chunks[idx])

    context = "\n\n".join(retrieved_chunks)

    # =====================================================
    # SAFETY CHECK (FIXED THRESHOLD)
    # =====================================================

    if len(context.strip()) < 50:
        print("\nNo relevant information found in document.\n")
        continue

    # =====================================================
    # PROMPT
    # =====================================================

    prompt = f"""
You are a professional equity research analyst.

Use ONLY the context below.

If answer is not found, say:
The document does not contain enough information.

CONTEXT:
{context}

QUESTION:
{query}
"""

    # =====================================================
    # AI RESPONSE (FIXED INSIDE LOOP)
    # =====================================================

    try:

        messages = [
            {"role": "system", "content": "You are a professional equity research analyst."}
        ]

        messages.extend(chat_history)

        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0
        )

        answer = response.choices[0].message.content

        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": answer})

        chat_history = chat_history[-20:]

        print("\nANSWER:\n")
        print(answer)
        print("\n" + "=" * 60 + "\n")

    except Exception as e:
        print("\nERROR:\n", e)