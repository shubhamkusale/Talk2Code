import os

from dotenv import load_dotenv

from app.ingestion.loading.git_api_fetch import GitHubAPIFetcher
from app.ingestion.chunking.chunking import Chunker
from app.ingestion.embedding.embed import Embedder


load_dotenv()
print("Imports done")

def index_repository(github_url: str):
    """
    Fetch a GitHub repository, chunk its files, create embeddings,
    and upload the resulting points to Qdrant.
    """

    github_token = os.getenv("GITHUB_TOKEN")

    # --------------------------------------------------------
    # 1. Fetch repository files
    # --------------------------------------------------------

    print("Fetching repository...")

    fetcher = GitHubAPIFetcher(
        github_url=github_url,
        token=github_token,
    )

    files = fetcher.fetch_all_files()

    print(f"Fetched {len(files)} files.")

    # --------------------------------------------------------
    # 2. Chunk repository files
    # --------------------------------------------------------

    print("Chunking files...")

    chunker = Chunker()

    all_chunks = []

    for filename, contents in files.items():

        if not contents.strip():
            continue

        chunks = chunker.make_points(
            {
                filename: contents
            }
        )

        all_chunks.extend(chunks)

    print(f"Created {len(all_chunks)} chunks.")

    # --------------------------------------------------------
    # 3. Create embeddings and Qdrant points
    # --------------------------------------------------------

    print("Creating embeddings...")

    embedder = Embedder()

    embedder.create_collection()

    points = embedder.create_points(all_chunks)

    print(f"Created {len(points)} Qdrant points.")

    # --------------------------------------------------------
    # 4. Upload points to Qdrant
    # --------------------------------------------------------

    print("Uploading to Qdrant...")

    embedder.upload(points)

    print("Repository indexing complete.")


if __name__ == "__main__":

    github_url = input("Enter GitHub repository URL: ").strip()

    index_repository(github_url)