import HP
import numpy as np
# generate pre-trained embedding


class Embedding:
    embedding = None

    @staticmethod
    def get_embedding():
        if Embedding.embedding is None:
            Embedding.embedding = Embedding.load_embedding()
        return Embedding.embedding

    @staticmethod
    def load_embedding():
        embedding_file = open(HP.embedding_file)
        embedding_map = {}
        for line in embedding_file:
            values = line.split()
            word = values[0]
            coef = np.asarray(values[1:], dtype='float32')
            embedding_map[word] = coef
        embedding_file.close()
        return embedding_map
