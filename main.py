from google import genai
from dotenv import load_dotenv
from bm25 import BM25
from vec_emb import VEC_EMB
import os

load_dotenv()

documentos = [
    "A política de reembolso da empresa garante devolução do dinheiro em até 7 dias úteis.",
    "O horário de atendimento do suporte técnico é de segunda a sexta, das 8h às 18h.",
    "Para resetar sua senha, clique em 'Esqueci minha senha' na tela de login e siga as instruções.",
    "A garantia dos produtos eletrônicos é de 1 ano contra defeitos de fabricação."
]

client = genai.Client(api_key=os.getenv("API_KEY"))

searcher_bm25 = BM25(documentos)
searcher_emb = VEC_EMB(client, documentos)

while True:
    question = input("Você: ").strip()

    if not question:
        print("Digite uma pergunta.")
        continue

    c_bm25 = searcher_bm25.rank_bm25(question)
    c_emb = searcher_emb.rank_vec_emb(question)

    c_final = []
    for i in range(len(documentos)):
        c_final.append((c_bm25[i] + c_emb[i], documentos[i]))

    c_final.sort(key=lambda x: x[0], reverse=True)
    contexto = "\n".join(
        doc for _, doc in c_final[:3]
    )

    prompt_contexto = f"""
    Você é um assistente virtual. Use o Contexto para responder à Pergunta.
    Se o contexto não for útil, responda "Não tenho dados suficientes para responder corretamente.".

    Contexto:
    {contexto}

    Pergunta:
    {question}
    """

    response = client.models.generate_content(
        model= "gemini-3.1-flash-lite",
        contents= prompt_contexto
    )

    print(f"Chat: {response.text}")