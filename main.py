import os
from pathlib import Path

from google import genai
from google.genai import types
from dotenv import load_dotenv

from bm25 import BM25
from vec_emb import VEC_EMB


def load_documents():
    """Load non-empty lines from text files in the docs directory."""
    documents = []

    for arq in Path("docs").glob("*.txt"):
        with open(arq, "r", encoding="utf-8") as f:
            for linha in f:
                linha = linha.strip()
                if linha:
                    documents.append(linha)

    return documents


def retrieve_context(question, documents, searcher_bm25, searcher_emb):
    """
    Retrieve the most relevant documents for a given question.

    Combines BM25 and embedding scores to produce a hybrid ranking.
    """
    # Already normalized scores
    bm25_scores = searcher_bm25.rank_bm25(question)
    embedding_scores = searcher_emb.rank_vec_emb(question)

    final_scores = []
    for i in range(len(documents)):
        final_scores.append(
            (
                0.5 * bm25_scores[i] + 0.5 * embedding_scores[i],
                documents[i],
            )  # Could change individual weights here
        )

    final_scores.sort(key=lambda x: x[0], reverse=True)

    if final_scores[0][0] < 0.6:
        return None

    context = "\n".join(doc for _, doc in final_scores[:2])

    return context


def generate_answer(client, config, question, context):
    """Send a prompt to Gemini and generate an answer."""
    prompt_context = f"""
    Você é um assistente virtual especialista e factual. Seu objetivo é responder à Pergunta do usuário baseando-se estritamente no Contexto fornecido.

    Diretrizes estritas:
    1. Baseie sua resposta APENAS nas informações contidas no "Contexto" abaixo.
    2. Não utilize nenhum conhecimento prévio ou externo ao texto fornecido.
    3. Se o contexto não contiver a resposta exata ou não for útil, responda rigorosamente com a frase: "Não tenho dados suficientes para responder corretamente." e nada mais.
    4. Seja direto, objetivo e evite suposições.

    ### Contexto:
    {context}

    ### Pergunta:
    {question}
    """

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite", contents=prompt_context, config=config
    )

    return response.text


def chat(client, config, documents, searcher_bm25, searcher_emb):
    """Run the interactive chat loop."""
    while True:
        question = input("Você: ").strip()

        if not question:
            print("Digite uma pergunta.")
            continue

        context = retrieve_context(question, documents, searcher_bm25, searcher_emb)

        if context is None:
            print("Não tenho dados suficientes para responder corretamente")
        else:
            response = generate_answer(client, config, question, context)
            print(f"Chat: {response}")


def main():
    """Initialize the system and start the chat loop."""
    load_dotenv()

    documents = load_documents()

    client = genai.Client(api_key=os.getenv("API_KEY"))

    config = types.GenerateContentConfig(temperature=0.2, top_p=0.9, top_k=40)

    searcher_bm25 = BM25(documents)
    searcher_emb = VEC_EMB(client, documents)

    chat(client, config, documents, searcher_bm25, searcher_emb)


if __name__ == "__main__":
    main()
