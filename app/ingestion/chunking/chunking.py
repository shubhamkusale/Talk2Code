from langchain_text_splitters import RecursiveCharacterTextSplitter


class Chunker:
    def __init__(self,chunk_size: int = 2000,chunk_overlap: int = 200,):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n","\n"," ","",],
        )

    def make_points(self, file: dict) -> list[dict]:
        """
        Receive a dictionary containing one file:
            {"filename.py": "file contents..."}

        Return Qdrant-ready points with an empty vector.
        """
        if not file or len(file) != 1:
            raise ValueError("file must contain exactly one filename and its contents")

        filename, contents = next(iter(file.items()))

        if not contents.strip():
            return []

        chunks = self.splitter.split_text(contents)

        points = []
        search_start = 0

        for chunk in chunks:
            # Find where this chunk occurs in the original file.
            start_pos = contents.find(chunk, search_start)

            if start_pos == -1:
                continue

            end_pos = start_pos + len(chunk)

            start_line = contents.count("\n", 0, start_pos) + 1
            end_line = contents.count("\n", 0, end_pos) + 1

            points.append(
                {
                    "vector": [],
                    "payload": {
                        "filename": filename,
                        "text": chunk,
                        "start_line": start_line,
                        "end_line": end_line,
                    },
                }
            )

            # Move forward while accounting for overlap.
            search_start = start_pos + 1

        return points