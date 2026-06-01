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
    raise ValueError(
        "OPENAI_API_KEY not found in .env file"
    )

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

# =========================================================
# DEBUGGING
# =========================================================

print(
    f"TOTAL EXTRACTED TEXT LENGTH: "
    f"{len(full_text)}"
)

# =========================================================
# SAVE EXTRACTED TEXT
# =========================================================

os.makedirs("data", exist_ok=True)

with open(
    "data/extracted_text.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(full_text)

print("Extracted text saved")

# =========================================================
# CREATE CHUNKS
# =========================================================

print("\nCreating chunks...\n")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_text(full_text)

# REMOVE USELESS SMALL CHUNKS
chunks = [
    chunk for chunk in chunks
    if len(chunk.strip()) > 100
]

print(f"Total chunks created: {len(chunks)}")

# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

print("\nLoading embedding model...\n")

model = SentenceTransformer(
    "BAAI/bge-small-en"
)

# =========================================================
# CREATE EMBEDDINGS
# =========================================================

print("\nGenerating embeddings...\n")

embeddings = model.encode(
    chunks,
    show_progress_bar=True
)

embeddings = np.array(
    embeddings
).astype("float32")

print("Embeddings created successfully")

# =========================================================
# CREATE FAISS INDEX
# =========================================================

print("\nCreating FAISS vector database...\n")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("FAISS database ready")

# =========================================================
# SAVE FAISS INDEX
# =========================================================

os.makedirs("vector_store", exist_ok=True)

faiss.write_index(
    index,
    "vector_store/financial_index.faiss"
)

print("FAISS index saved")

# =========================================================
# AI QUESTION LOOP
# =========================================================

print("\n===================================")
print("AI EQUITY RESEARCH ASSISTANT READY")
print("Type 'exit' to stop")
print("===================================\n")

chat_history = []

while True:

    query = input("Question: ")

    # EXIT CONDITION
    if query.lower() == "exit":

        print("\nExiting assistant...\n")

        break

    # =====================================================
    # CREATE QUERY EMBEDDING
    # =====================================================

    query_embedding = model.encode([query])

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    # =====================================================
    # SEARCH VECTOR DATABASE
    # =====================================================

    D, I = index.search(
        query_embedding,
        k=12
    )

    # =====================================================
    # BUILD CONTEXT
    # =====================================================

    retrieved_chunks = []

    for idx in I[0]:

        chunk = chunks[idx]

        if len(chunk.strip()) > 100:

            retrieved_chunks.append(chunk)

    context = "\n\n".join(retrieved_chunks)

    # =====================================================
    # SAFETY CHECK
    # =====================================================

    if len(context.strip()) < 200:

        print(
            "\nThe document does not contain enough information.\n"
        )

        continue

    # =====================================================
    # CREATE PROMPT
    # =====================================================

    prompt = f"""
You are a professional equity research analyst.

Use ONLY the information provided in the context.

If the answer cannot be found in the context,
reply exactly with:

The document does not contain enough information.

Be concise, accurate, and professional.

========================
CONTEXT
========================

{context}

========================
QUESTION
========================

{query}
"""

    # =====================================================
# GENERATE AI RESPONSE
# =====================================================

try:

    messages = [
        {
            "role": "system",
            "content": "You are a professional equity research analyst."
        }
    ]

    messages.extend(chat_history)

    messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    response = client.chat.completions.create(

        model="gpt-4.1-mini",

        messages=messages,

        temperature=0
    )

    answer = response.choices[0].message.content

    chat_history.append(
        {
            "role": "user",
            "content": query
        }
    )

    chat_history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    # Keep only last 10 question-answer pairs
    chat_history = chat_history[-20:]

    print("\nANSWER:\n")

    print(answer)

    print("\n" + "=" * 60 + "\n")

except Exception as e:

    print("\nERROR GENERATING RESPONSE:\n")

    print(e)