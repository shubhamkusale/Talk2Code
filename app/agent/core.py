import os
from dotenv import load_dotenv
from groq import Groq
from app.ingestion.embedding.embed import Embedder

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
TOP_K = 5

_embedder = None
_llm = None


def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def _get_llm() -> Groq:
    global _llm
    if _llm is None:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY missing. Put it in the .env file in the project root.")
        _llm = Groq(api_key=key)
    return _llm


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    emb = _get_embedder()
    vector = emb.embedding_model.embed_query(question)
    result = emb.client.query_points(
        collection_name=emb.collection_name,
        query=vector,
        limit=top_k,
        with_payload=True,
    )
    return [p.payload for p in result.points]


def ask_agent(question: str) -> str:
    chunks = retrieve(question)
    if not chunks:
        return "I found nothing indexed yet. Index a repository first."

    context = "\n\n".join(
        f"[{c['filename']} lines {c['start_line']}-{c['end_line']}]\n{c['text']}"
        for c in chunks
    )

    system = (
        "You are Talk2Code, an assistant that explains a codebase. "
        "Answer using ONLY the code excerpts provided. Cite sources as "
        "file name and line range. If the excerpts don't contain the answer, say so."
    )
    response = _get_llm().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Code excerpts:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content