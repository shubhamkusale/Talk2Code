"""
github_fetch.py

Handles acquisition of a target GitHub repository for downstream processing
(chunking, embedding, and RAG-based interaction by the AI agent).

Usage (from elsewhere in the pipeline):

    from app.ingestion.loader.github_fetch import Cloner

    cloner = Cloner(github_url)
    repo_path = cloner.clone()
    cloner.list_files()
"""

import os
import shutil
import subprocess


class Cloner:
    """Handles shallow-cloning of a GitHub repo to a local destination,
    keeping the clone outside this project's own repo so it doesn't
    pollute version control or get mistaken for project source.
    """

    def __init__(self, github_url: str, dest_dir: str = None):
        self.github_url = github_url
        self.dest_dir = dest_dir or self._derive_dest_dir(github_url)

    @staticmethod
    def _derive_dest_dir(github_url: str) -> str:
        """Derive a destination path from the repo URL, placed outside this
        project's own git repo (e.g. E:\\Talk2Code\\<repo_name>) rather than
        relative to wherever this module happens to be imported/run from.
        """
        name = github_url.rstrip("/").split("/")[-1]
        if name.endswith(".git"):
            name = name[:-4]

        project_root = Cloner._find_project_root()
        return os.path.join(project_root, name)

    @staticmethod
    def _find_project_root(start: str = None) -> str:
        """Walk upward from `start` (default: this file's location) to find
        the repo's .git folder, then return the folder ONE LEVEL ABOVE it,
        so cloned repos never end up nested inside this project.
        """
        path = os.path.abspath(start or __file__)
        if os.path.isfile(path):
            path = os.path.dirname(path)

        while True:
            if os.path.isdir(os.path.join(path, ".git")):
                return os.path.dirname(path)
            parent = os.path.dirname(path)
            if parent == path:
                # No .git found; fall back to current working directory.
                return os.getcwd()
            path = parent

    def _remove_existing(self) -> None:
        if os.path.exists(self.dest_dir):
            print(f"Removing existing directory: {self.dest_dir}")
            shutil.rmtree(self.dest_dir)

    def _ensure_parent_dir(self) -> None:
        os.makedirs(os.path.dirname(self.dest_dir) or ".", exist_ok=True)

    def clone(self) -> str:
        """Shallow-clone the repo, replacing any existing copy at dest_dir."""
        self._remove_existing()
        self._ensure_parent_dir()

        print(f"Cloning {self.github_url} into {self.dest_dir} (depth=1)...")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", self.github_url, self.dest_dir],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"git clone failed: {result.stderr.strip()}")

        print(f"Clone complete: {self.dest_dir}")
        return self.dest_dir

    def list_files(self) -> list:
        """Print and return the top-level contents of the cloned repo (like `ls`)."""
        entries = sorted(os.listdir(self.dest_dir))
        print(f"\nContents of {self.dest_dir}:")
        for entry in entries:
            full_path = os.path.join(self.dest_dir, entry)
            marker = "/" if os.path.isdir(full_path) else ""
            print(f"  {entry}{marker}")
        return entries