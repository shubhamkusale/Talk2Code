"""
github_api_fetch.py

Fetches repository file contents directly via the GitHub REST API instead
of doing a full `git clone`. Useful when you only need specific files, or
want to avoid disk I/O for a shallow clone, before handing content off to
the chunking/embedding stage of the RAG pipeline.

GitHub's API returns blob content base64-encoded, so this module decodes
it with Python's built-in `base64` library.

Usage:

    from app.ingestion.loader.github_api_fetch import GitHubAPIFetcher

    fetcher = GitHubAPIFetcher(github_url, token=os.getenv("GITHUB_TOKEN"))
    files = fetcher.fetch_all_files()   # {relative_path: file_content_str}
"""
from dotenv import load_dotenv
load_dotenv()

import base64
import os
import re
from typing import Dict, List, Optional

import requests

API_ROOT = "https://api.github.com"

# Extensions we generally don't want to spend API calls / embeddings on.
DEFAULT_IGNORED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
    ".pdf", ".zip", ".tar", ".gz", ".rar",
    ".lock", ".log",
    ".woff", ".woff2", ".ttf", ".eot",
    ".exe", ".dll", ".so", ".bin",
    ".mp3", ".mp4", ".mov", ".avi",
}


class GitHubAPIFetcher:
    """Fetches a repo's file tree and blob contents via the GitHub API."""

    def __init__(self, github_url: str, token: Optional[str] = None, branch: Optional[str] = None):
        self.owner, self.repo = self._parse_owner_repo(github_url)
        self.token = token
        self.branch = branch or self._get_default_branch()

    @staticmethod
    def _parse_owner_repo(github_url: str) -> tuple:
        """Extract (owner, repo) from a GitHub URL, e.g.
        https://github.com/owner/repo.git -> ("owner", "repo")
        """
        match = re.search(r"github\.com/([^/]+)/([^/.]+)", github_url)
        if not match:
            raise ValueError(f"Could not parse owner/repo from URL: {github_url}")
        return match.group(1), match.group(2)

    def _headers(self) -> dict:
        headers = {"Accept": "application/vnd.github+json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _get_default_branch(self) -> str:
        url = f"{API_ROOT}/repos/{self.owner}/{self.repo}"
        response = requests.get(url, headers=self._headers())
        response.raise_for_status()
        return response.json()["default_branch"]

    def get_file_tree(self) -> list:
        """Return the full recursive file tree (list of blob metadata dicts)
        for the repo's current branch, via the Git Trees API.
        """
        url = f"{API_ROOT}/repos/{self.owner}/{self.repo}/git/trees/{self.branch}"
        response = requests.get(url, headers=self._headers(), params={"recursive": "1"})
        response.raise_for_status()
        tree = response.json().get("tree", [])
        return [entry for entry in tree if entry["type"] == "blob"]

    @staticmethod
    def filter_tree(
        tree: List[dict],
        ignored_extensions: Optional[set] = None,
    ) -> List[dict]:
        """Drop blobs whose file extension is in `ignored_extensions`
        (defaults to DEFAULT_IGNORED_EXTENSIONS) so we don't waste API
        calls / embeddings on binaries, lockfiles, media, etc.
        """
        ignored = ignored_extensions if ignored_extensions is not None else DEFAULT_IGNORED_EXTENSIONS
        filtered = []
        for entry in tree:
            _, ext = os.path.splitext(entry["path"])
            if ext.lower() not in ignored:
                filtered.append(entry)
        return filtered

    def fetch_blob_content(self, sha: str) -> str:
        """Fetch and base64-decode a single blob's content by its SHA."""
        url = f"{API_ROOT}/repos/{self.owner}/{self.repo}/git/blobs/{sha}"
        response = requests.get(url, headers=self._headers())
        response.raise_for_status()
        blob = response.json()

        if blob.get("encoding") != "base64":
            raise ValueError(f"Unexpected encoding for blob {sha}: {blob.get('encoding')}")

        decoded_bytes = base64.b64decode(blob["content"])
        try:
            return decoded_bytes.decode("utf-8")
        except UnicodeDecodeError:
            # Binary file (image, etc.) — caller can decide how to handle this.
            return ""

    def fetch_all_files(self, ignored_extensions: Optional[set] = None) -> Dict[str, str]:
        """Fetch every non-ignored file in the repo, returning
        {relative_path: content}. Binary files that slip through the
        extension filter are included with empty string content.
        """
        tree = self.filter_tree(self.get_file_tree(), ignored_extensions)
        files = {}
        for entry in tree:
            print(f"Fetching {entry['path']}...")
            files[entry["path"]] = self.fetch_blob_content(entry["sha"])
        print(f"Fetched {len(files)} files from {self.owner}/{self.repo}@{self.branch}")
        return files


def main():
    """Temporary CLI entry point for manually testing this module in the
    terminal. Remove once this is wired into the actual ingestion pipeline.
    """
    github_url = input("Enter GitHub repo URL: ").strip()
    if not github_url:
        print("Error: GitHub URL cannot be empty.")
        return

    token = os.getenv("GITHUB_TOKEN")  # optional, raises rate limit if set

    try:
        fetcher = GitHubAPIFetcher(github_url, token=token)
        files = fetcher.fetch_all_files()
    except (ValueError, requests.HTTPError) as e:
        print(f"Error: {e}")
        return

    print("\nFiles fetched:")
    for path, content in files.items():
        preview = content[:60].replace("\n", " ") if content else "(binary or empty)"
        print(f"  {path}  ->  {preview}...")


if __name__ == "__main__":
    main()