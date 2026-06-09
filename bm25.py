import math

class BM25:
    def __init__(self, documentos, k1 = 1.5, b = 0.75):
        self.documentos_originais = documentos
        self.k1 = k1
        self.b = b
        self.N = len(documentos)

        self.corpus_tokenizado = [self._tokenizar(doc) for doc in documentos]
        self.comprimentos_docs = [len(doc) for doc in self.corpus_tokenizado]
        self.comprimento_medio = sum(self.comprimentos_docs) / self.N if self.N > 0 else 0

        self.idf = self._calcular_idf()

    def _tokenizar(self, texto):
        pontuacao = [".", ",", "'", '"', "(", ")", "-", "!", "?", ";"]
        txt = texto.lower()
        for char in pontuacao:
            txt = txt.replace(char, "")
        return txt.split()
    
    def _calcular_idf(self):
        todas_palavras = set(palavra for doc in self.corpus_tokenizado for palavra in doc)
        idf = {}
        for palavra in todas_palavras:
            doc_count = sum(1 for doc in self.corpus_tokenizado if palavra in doc)

            idf[palavra] = math.log((self.N - doc_count + 0.5) / (doc_count + 0.5) + 1)
        
        return idf

    def _pontuar_documento(self, query_tokenizada, doc_tokenizado, doc_len):
        score = 0.0
        for palavra in query_tokenizada:
            if palavra not in self.idf:
                continue
        
            tf = doc_tokenizado.count(palavra)
            numerador = tf * (self.k1 + 1)
            denominador = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.comprimento_medio))
            
            score += self.idf[palavra] * (numerador / denominador)
        return score
    
    def rank_bm25(self, pergunta):
        query_tokenizada = self._tokenizar(pergunta)

        scores = []
        for i, doc_tokenizado in enumerate(self.corpus_tokenizado):
            score = self._pontuar_documento(query_tokenizada, doc_tokenizado, self.comprimentos_docs[i])
            scores.append(score)

        return scores
