# mcp/git_client.py
import os
import base64
import httpx
from typing import List

GIT_MCP_ENDPOINT = os.getenv("GIT_MCP_ENDPOINT", "http://localhost:9000/git-mcp")
GIT_TOKEN = os.getenv("GIT_TOKEN")

class MCPGitClient:
    """
    Adapter over a Git MCP tool or HTTP service.
    For now, keep method signatures simple for LLM to reason on top.
    """

    def __init__(self, repo: str):
        self.repo = repo
        self.client = httpx.Client(
            base_url=GIT_MCP_ENDPOINT,
            headers={"Authorization": f"Bearer {GIT_TOKEN}"} if GIT_TOKEN else {}
        )

    def get_file(self, path: str) -> str:
        resp = self.client.post("/file", json={"repo": self.repo, "path": path})
        resp.raise_for_status()
        data = resp.json()

        # Common cases: top-level 'content' possibly with 'encoding'
        if isinstance(data, dict):
            if "content" in data:
                content = data["content"]
                if data.get("encoding") == "base64":
                    try:
                        return base64.b64decode(content).decode("utf-8")
                    except Exception as e:
                        raise RuntimeError(f"failed to decode base64 content from git-mcp: {e}")
                return content

            # Nested shapes: {"file": {"content":...}} or {"data": {...}}
            for k in ("file", "data", "result"):
                v = data.get(k)
                if isinstance(v, dict) and "content" in v:
                    content = v["content"]
                    if v.get("encoding") == "base64":
                        try:
                            return base64.b64decode(content).decode("utf-8")
                        except Exception as e:
                            raise RuntimeError(f"failed to decode base64 content from git-mcp: {e}")
                    return content

            # Sometimes the endpoint returns a list under 'files'
            files = data.get("files")
            if isinstance(files, list) and files:
                first = files[0]
                if isinstance(first, dict) and "content" in first:
                    content = first["content"]
                    if first.get("encoding") == "base64":
                        try:
                            return base64.b64decode(content).decode("utf-8")
                        except Exception as e:
                            raise RuntimeError(f"failed to decode base64 content from git-mcp: {e}")
                    return content

        # Fall back to returning raw text if JSON shape is unexpected but there is body text
        text = resp.text
        if text:
            return text

        raise RuntimeError(f"Unexpected response from git-mcp when fetching '{path}': {data}")

    def list_java_files(self, base_path: str = "src/main/java") -> List[str]:
        resp = self.client.post("/list", json={"repo": self.repo, "base_path": base_path, "ext": ".java"})
        resp.raise_for_status()
        return resp.json()["files"]

    def get_pr_diff(self, pr_number: int) -> str:
        resp = self.client.post("/pr-diff", json={"repo": self.repo, "pr_number": pr_number})
        resp.raise_for_status()
        return resp.json()["diff"]
