import math

class BM25:
    def __init__(self, documents, k1 = 1.5, b = 0.75):
        self.original_documents = documents
        self.k1 = k1
        self.b = b
        self.N = len(documents)

        self.tokenized_corpus = [self._tokenize(doc) for doc in documents]
        self.docs_len = [len(doc) for doc in self.tokenized_corpus]
        self.md_len = sum(self.docs_len) / self.N if self.N > 0 else 0

        self.idf = self._calculate_idf()

    def _tokenize(self, text):
        punctuation = [".", ",", "'", '"', "(", ")", "-", "!", "?", ";"]
        txt = text.lower()
        for char in punctuation:
            txt = txt.replace(char, "")
        return txt.split()
    
    def _calculate_idf(self):
        words = set(word for doc in self.tokenized_corpus for word in doc)
        idf = {}
        for word in words:
            doc_count = sum(1 for doc in self.tokenized_corpus if word in doc)

            idf[word] = math.log((self.N - doc_count + 0.5) / (doc_count + 0.5) + 1)
        
        return idf

    def _score_document(self, tokenized_query, tokenized_doc, doc_len):
        score = 0.0
        for word in tokenized_query:
            if word not in self.idf:
                continue
        
            tf = tokenized_doc.count(word)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.md_len))
            
            score += self.idf[word] * (numerator / denominator)
        return score
    
    def rank_bm25(self, question):
        tokenized_query = self._tokenize(question)

        scores = []
        for i, tokenized_doc in enumerate(self.tokenized_corpus):
            score = self._score_document(tokenized_query, tokenized_doc, self.docs_len[i])
            scores.append(score)
        
        max_score = max(scores)
        for i, score in enumerate(scores):
            scores[i] = score / max_score

        return scores
