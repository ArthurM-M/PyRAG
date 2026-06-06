from google import genai
from dotenv import load_dotenv
from bm25 import BM25
import os

load_dotenv()

documentos = [
    "A política de reembolso da empresa garante devolução do dinheiro em até 7 dias úteis.",
    "O horário de atendimento do suporte técnico é de segunda a sexta, das 8h às 18h.",
    "Para resetar sua senha, clique em 'Esqueci minha senha' na tela de login e siga as instruções.",
    "A garantia dos produtos eletrônicos é de 1 ano contra defeitos de fabricação."
]

searcher = BM25(documentos)

client = genai.Client(api_key=os.getenv("API_KEY"))

while True:
    question = input("Você: ")

    contexto = searcher.buscar_melhor_contexto(question)

    prompt_contexto = f"""
    Você é um assistente virtual. Use o Contexto para responder à Pergunta.
    Se o contexto não for útil, responda "Não tenho dados suficientes para responder corretamente.".

    Contexto:
    {contexto}

    Pergunta:
    {question}
    """

    print(f"Prompt com contexto: {prompt_contexto}")

    response = client.models.generate_content(
        model= "gemini-3.1-flash-lite",
        contents= prompt_contexto
    )

    print(f"Chat: {response.text}")