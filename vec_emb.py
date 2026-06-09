from google import genai
import numpy as np

class VEC_EMB:
    def __init__(self, client, documents):
        self.client = client
        self.documents = documents
        self.doc_embeddings = []

        for doc in documents:
            emb = client.models.embed_content(
                model="gemini-embedding-001",
                contents=doc
            )
            self.doc_embeddings.append(np.array(emb.embeddings[0].values))
    
    def _cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def rank_vec_emb(self, question):
        emb = self.client.models.embed_content(
            model="gemini-embedding-001",
            contents=question
        )

        v_question = np.array(emb.embeddings[0].values)

        scores = []

        for i, v_doc in enumerate(self.doc_embeddings):
            score = self._cosine_similarity(v_question, v_doc)
            scores.append(score)
        
        return scores