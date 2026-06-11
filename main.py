import os
from pathlib import Path

from google import genai
from dotenv import load_dotenv

from bm25 import BM25
from vec_emb import VEC_EMB


def load_documents():
    documents = []

    for arq in Path("docs").glob("*.txt"):
        with open(arq, "r", encoding="utf-8") as f:
            for linha in f:
                linha = linha.strip()
                if linha:
                    documents.append(linha)

    return documents


def retrieve_context(question, documents, searcher_bm25, searcher_emb):
    c_bm25 = searcher_bm25.rank_bm25(question)
    c_emb = searcher_emb.rank_vec_emb(question)

    c_final = []
    for i in range(len(documents)):
        c_final.append((c_bm25[i] + c_emb[i], documents[i]))

    c_final.sort(key=lambda x: x[0], reverse=True)

    context = "\n".join(doc for _, doc in c_final[:2])

    return context


def generate_answer(client, question, context):
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

    return response.text


def chat(client, documents, searcher_bm25, searcher_emb):
    while True:
        question = input("Você: ").strip()

        if not question:
            print("Digite uma pergunta.")
            continue
        
        context = retrieve_context(question, documents, searcher_bm25, searcher_emb)

        response = generate_answer(client, question, context)
    
        print(f"Chat: {response}")

    
def main():
    load_dotenv()

    documents = load_documents()

    client = genai.Client(api_key=os.getenv("API_KEY"))

    searcher_bm25 = BM25(documents)
    searcher_emb = VEC_EMB(client, documents)

    chat(client, documents, searcher_bm25, searcher_emb)


if __name__ == "__main__":
    main()