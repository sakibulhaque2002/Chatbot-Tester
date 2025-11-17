import numpy as np
from openai import OpenAI
from config import EMBEDDING_MODEL

# Two vLLM servers
embedding_client = OpenAI(base_url="http://localhost:8001/v1", api_key="EMPTY")

def get_embeddings(texts):
    response = embedding_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return [np.array(r.embedding, dtype=np.float16) for r in response.data]
