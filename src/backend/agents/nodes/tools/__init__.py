from backend.agents.utils.helper import _get_secret


COHERE_EMBED_MODEL     = "embed-english-v3.0"
EMBED_INPUT_TYPE_DOC   = "search_document"
EMBED_INPUT_TYPE_QUERY = "search_query"
EMBED_DIM              = 1024

# Minimum cosine similarity to include a chunk in results.
# IndexFlatIP on L2-normalised vectors = cosine similarity in [-1, 1].
# Chunks scoring below 0.30 are almost certainly off-topic.
MIN_SCORE = 0.30

# Number of chunks to retrieve per query (final after fusion)
TOP_K = 5


__all__= [
    "_get_secret", "COHERE_EMBED_MODEL", "EMBED_INPUT_TYPE_DOC", "EMBED_INPUT_TYPE_QUERY", "EMBED_DIM",
    "MIN_SCORE", "TOP_K"
    ]