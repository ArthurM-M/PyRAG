from google import genai
from dotenv import load_dotenv
from pathlib import Path
from bm25 import BM25
from vec_emb import VEC_EMB
import os

load_dotenv()

documents = []

for arq in Path("docs").glob("*.txt"):
    with open(arq, "r", encoding="utf-8") as f:
        documents.append(f.read())

client = genai.Client(api_key=os.getenv("API_KEY"))

searcher_bm25 = BM25(documents)
searcher_emb = VEC_EMB(client, documents)

while True:
    question = input("Você: ").strip()

    if not question:
        print("Digite uma pergunta.")
        continue

    c_bm25 = searcher_bm25.rank_bm25(question)
    c_emb = searcher_emb.rank_vec_emb(question)

    c_final = []
    for i in range(len(documents)):
        c_final.append((c_bm25[i] + c_emb[i], documents[i]))

    c_final.sort(key=lambda x: x[0], reverse=True)
    context = "\n".join(
        doc for _, doc in c_final[:3]
    )

    prompt_contexto = f"""
    Você é um assistente virtual. Use o Contexto para responder à Pergunta.
    Se o contexto não for útil, responda "Não tenho dados suficientes para responder corretamente.".

    Contexto:
    {context}

    Pergunta:
    {question}
    """

    response = client.models.generate_content(
        model= "gemini-3.1-flash-lite",
        contents= prompt_contexto
    )

    print(f"Chat: {response.text}")