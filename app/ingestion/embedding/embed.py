import os
from uuid import uuid4

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


load_dotenv()


class Embedder:
    """Create embeddings for chunks and store them in Qdrant."""

    def __init__(
        self,
        collection_name: str = "knowledge",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.collection_name = collection_name

        # --------------------------------------------------------
        # Qdrant
        # --------------------------------------------------------

        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not qdrant_url or not qdrant_api_key:
            raise ValueError(
                "QDRANT_URL and QDRANT_API_KEY must be set in .env"
            )

        self.client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )

        # --------------------------------------------------------
        # Hugging Face Embedding Model
        # --------------------------------------------------------

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={
                "device": "cpu",
            },
            encode_kwargs={
                "normalize_embeddings": True,
            },
        )

        # all-MiniLM-L6-v2 produces 384-dimensional embeddings
        self.embedding_size = 384

    def create_collection(self):
        """Create the Qdrant collection if it doesn't exist."""

        if self.client.collection_exists(self.collection_name):
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedding_size,
                distance=Distance.COSINE,
            ),
        )

    def create_points(self, chunks: list[dict]) -> list[PointStruct]:
        """
        Create Qdrant points from already-created chunks.

        Expected chunk format:

        {
            "vector": [],
            "payload": {
                "filename": "...",
                "text": "...",
                "start_line": 1,
                "end_line": 50
            }
        }
        """

        if not chunks:
            return []

        # Extract the text from every chunk
        texts = [
            chunk["payload"]["text"]
            for chunk in chunks
        ]

        # One embedding is generated for each chunk
        embeddings = self.embedding_model.embed_documents(texts)

        points = []

        for chunk, embedding in zip(chunks, embeddings):

            point = PointStruct(
                id=str(uuid4()),
                vector=embedding,
                payload=chunk["payload"],
            )

            points.append(point)

        return points

    def upload(self, points: list[PointStruct]):
        """Upload points to Qdrant."""

        if not points:
            print("No points to upload.")
            return

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        print(f"Uploaded {len(points)} points to Qdrant.")