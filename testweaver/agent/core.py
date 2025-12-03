# agent/core.py
import pathlib
from typing import Optional

from ..llm.client import LLMClient
from ..memory.short_term import ShortTermMemory
from ..rag.index import RAGIndex
from ..mcp.git_client import MCPGitClient

class TestWeaverAgent:
    def __init__(self, session_id: str, rag_index: RAGIndex, short_term: ShortTermMemory, repo: str):
        self.session_id = session_id
        self.rag_index = rag_index
        self.short_term = short_term
        self.git = MCPGitClient(repo)
        self.llm = LLMClient()
        # Base directory where this file is located
        BASE_DIR = pathlib.Path(__file__).resolve().parent.parent  # one level up to testweaver/

        PROMPTS_DIR = BASE_DIR / "prompts"

        self.system_prompt = (PROMPTS_DIR / "system_agent.md").read_text(encoding="utf-8")
        self.test_prompt = (PROMPTS_DIR / "test_generation.md").read_text(encoding="utf-8")

    def _build_messages(self, user_message: str, task_context: Optional[str] = None):
        history = self.short_term.get_history(self.session_id)
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(history)  # prior turns
        if task_context:
            messages.append({"role": "user", "content": f"<context>\n{task_context}\n</context>"})
        messages.append({"role": "user", "content": user_message})
        return messages

    def chat(self, user_message: str, query_for_rag: Optional[str] = None) -> str:
        # retrieve RAG context if query provided
        task_context = ""
        if query_for_rag:
            task_context = self.rag_index.retrieve_context(query_for_rag, top_k=5)

        messages = self._build_messages(user_message, task_context=task_context)
        response = self.llm.chat(messages)
        self.short_term.append(self.session_id, "user", user_message)
        self.short_term.append(self.session_id, "assistant", response)
        return response

    def generate_tests_for_file(self, service_path: str, extra_instructions: str = "") -> str:
        java_source = self.git.get_file(service_path)

        # 2. Build a generic RAG query based on user input + file name
    #    - primary: whatever the user asked for in extra_instructions
    #    - secondary: the Java class name from the path
    #    - fallback: a neutral default
        class_name = service_path.split("/")[-1].replace(".java", "")
        rag_query_parts = []

        if extra_instructions:
            rag_query_parts.append(extra_instructions)

        rag_query_parts.append(class_name)

        rag_query = " ".join(rag_query_parts).strip() or "test generation for service"

        # Optional debug log (requires logger import)
        # logger.debug("RAG query for generate_tests_for_file: %r", rag_query)

        # 3. Retrieve context from RAG using this query
        rag_context = self.rag_index.retrieve_context(rag_query, top_k=5)

        # 4. Build the LLM prompt
        user_msg = f"""
You must generate JUnit tests for the following Java file:

<source_path>{service_path}</source_path>

<source_code>
{java_source}
</source_code>

<context_from_docs>
{rag_context}
</context_from_docs>

Additional instructions from the user:
{extra_instructions}

First, think if any business logic or requirements are unclear.
If unclear, ask clarifying questions instead of directly generating tests.
If clear, output ONLY a compilable Java test class.
"""

        # 5. Call LLM with accumulated history + test-generation task prompt
        messages = self._build_messages(user_msg, task_context=self.test_prompt)
        response = self.llm.chat(messages)

        # 6. Update short-term memory
        self.short_term.append(self.session_id, "user", user_msg)
        self.short_term.append(self.session_id, "assistant", response)

        return response