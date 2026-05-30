"""
Embedding Engine Module - Sentence Transformers Integration
============================================================

Provides semantic embeddings using the sentence-transformers library
(all-MiniLM-L6-v2 model). Gracefully falls back if sentence-transformers
is not installed, allowing lightweight deployments without the heavy dependency.

Model: all-MiniLM-L6-v2
  - Dimensions: 384
  - Size: ~80MB
  - Speed: Fast (suitable for real-time inference)
  - Quality: Excellent for semantic similarity tasks

Usage:
    engine = EmbeddingEngine()
    if engine.is_available():
        embeddings = engine.encode(["hello world", "how are you"])
        scores = engine.similarity(query_emb, corpus_embs)
"""

import numpy as np


class EmbeddingEngine:
    """
    Semantic embedding engine using sentence-transformers.

    Provides encode() and similarity() methods for computing semantic
    similarity between texts. Gracefully handles the case where
    sentence-transformers is not installed.

    Attributes:
        available (bool): Whether sentence-transformers is loaded and ready.
        model: The SentenceTransformer model instance (or None).
        model_name (str): Name of the model being used.
        embedding_dim (int): Dimensionality of the embeddings (384 for MiniLM).
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the EmbeddingEngine.

        Attempts to load the sentence-transformers library and the specified model.
        If the library is not installed, sets self.available = False and prints
        a warning. The class remains instantiable but encode/similarity will
        raise RuntimeError.

        Args:
            model_name (str): The sentence-transformers model to load.
                Default: "all-MiniLM-L6-v2" (384-dim, fast, good quality).
        """
        self.model_name = model_name
        self.model = None
        self.available = False
        self.embedding_dim = None

        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
            self.available = True

            # Get embedding dimensions from the model
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

            print(f"[EmbeddingEngine] Model loaded: {model_name}")
            print(f"[EmbeddingEngine] Embedding dimensions: {self.embedding_dim}")

        except ImportError:
            self.available = False
            print(
                "[EmbeddingEngine] WARNING: sentence-transformers not installed. "
                "Embeddings will not be available. Install with: "
                "pip install sentence-transformers"
            )
        except Exception as e:
            self.available = False
            print(
                f"[EmbeddingEngine] WARNING: Failed to load model '{model_name}': {e}. "
                "Falling back to TF-IDF mode."
            )

    def encode(self, texts):
        """
        Encode a list of texts into dense embeddings.

        Args:
            texts (list of str): The texts to encode.

        Returns:
            numpy.ndarray: Array of shape (n, embedding_dim) containing
                the embeddings for each input text.

        Raises:
            RuntimeError: If sentence-transformers is not available.
        """
        if not self.available:
            raise RuntimeError(
                "EmbeddingEngine is not available. "
                "Install sentence-transformers: pip install sentence-transformers"
            )

        # Encode texts using the model
        # convert_to_numpy=True ensures we get a numpy array
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
        )

        return embeddings

    def similarity(self, query_embedding, corpus_embeddings):
        """
        Compute cosine similarity between a query embedding and corpus embeddings.

        Since embeddings are L2-normalized during encoding, cosine similarity
        reduces to a simple dot product.

        Args:
            query_embedding (numpy.ndarray): Shape (embedding_dim,) or (1, embedding_dim).
                The query vector to compare against the corpus.
            corpus_embeddings (numpy.ndarray): Shape (n, embedding_dim).
                The corpus of pre-computed embeddings to search.

        Returns:
            numpy.ndarray: 1D array of shape (n,) containing similarity scores
                between the query and each corpus entry. Scores range from -1 to 1
                (though typically 0 to 1 for natural language).
        """
        if not self.available:
            raise RuntimeError(
                "EmbeddingEngine is not available. "
                "Install sentence-transformers: pip install sentence-transformers"
            )

        # Ensure query is 2D for matrix multiplication
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Cosine similarity via dot product (embeddings are already L2-normalized)
        similarities = np.dot(query_embedding, corpus_embeddings.T).flatten()

        return similarities

    def is_available(self):
        """
        Check whether the embedding engine is available and ready for use.

        Returns:
            bool: True if sentence-transformers is installed and the model
                is loaded successfully. False otherwise.
        """
        return self.available
