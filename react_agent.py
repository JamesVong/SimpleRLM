import os
import re
import io
import time
import textwrap
import contextlib
from typing import Any, List, Dict

from dotenv import load_dotenv
import tiktoken
from rank_bm25 import BM25Okapi
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

MODEL_NAME = "nvidia/nemotron-3-super-120b-a12b"
TEMPERATURE = 0.9
TOP_P = 0.95

try:
    with open("react_system_prompt.txt", "r") as f:
        REACT_SYSTEM_PROMPT = f.read()
except FileNotFoundError:
    REACT_SYSTEM_PROMPT = "Error: react_system_prompt.txt not found."


class ReActAgent:
    """
    CodeAct agent following the paper's Algorithm 2 / CodeAct baseline.

    Key differences from RLM:
      - The full user prompt (context) is fed INTO the LLM's message history,
        not offloaded to a REPL variable. (Algorithm 2, Flaw #1 — intentionally
        the weaker design for comparison.)
      - BM25 retriever indexes the context messages for SEARCH() queries.
      - Terminates with "ANSWER: [text]", not FINAL()/FINAL_VAR().
      - No sub-LLM calls — code is only for computation/verification.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY not found.")

        self.llm = ChatNVIDIA(
            model=MODEL_NAME,
            nvidia_api_key=self.api_key,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )

        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")

        self.context: List[Dict[str, Any]] = []
        self._bm25 = None
        self._bm25_docs = []

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        try:
            return len(self.tokenizer.encode(str(text)))
        except Exception:
            return len(str(text)) // 2

    def _build_bm25_index(self):
        """Build a BM25 index over the context messages."""
        self._bm25_docs = []
        tokenized = []
        for i, msg in enumerate(self.context):
            content = msg.get("content", "")
            if not content.strip():
                continue
            self._bm25_docs.append((i, msg))
            tokenized.append(content.lower().split())
        if tokenized:
            self._bm25 = BM25Okapi(tokenized)
        else:
            self._bm25 = None

    def _search(self, query: str, top_k: int = 5) -> str:
        """BM25 search over context messages."""
        if not self._bm25 or not self._bm25_docs:
            return "No documents indexed for search."

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        ranked = sorted(
            zip(scores, self._bm25_docs),
            key=lambda x: -x[0],
        )[:top_k]

        results = []
        for score, (idx, msg) in ranked:
            if score <= 0:
                continue
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            preview = content[:500]
            results.append(
                f"[Message {idx}] (role={role}, score={score:.2f})\n{preview}"
                + ("..." if len(content) > 500 else "")
            )

        if not results:
            return "No relevant results found."
        return "\n\n---\n\n".join(results)

    def _parse_answer(self, content: str):
        """Parse ANSWER: from model output. Returns answer text or None."""
        match = re.search(r"ANSWER:\s*(.+)", content, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            # Trim trailing code fences or artifacts
            answer = re.split(r"\n```", answer)[0].strip()
            if answer:
                return answer
        return None

    def add_to_context(self, role: str, content: str):
        self.context.append({
            "role": role,
            "content": content,
            "tokens": self._count_tokens(content),
            "timestamp": time.time(),
        })

    def _build_context_summary(self) -> str:
        """
        Build a summary of the context to include in the LLM's message history.
        Per Algorithm 2: the prompt P goes into the LLM context window directly.
        For very large contexts, we include metadata + note about SEARCH.
        """
        total_tokens = sum(m.get("tokens", 0) for m in self.context)
        msg_count = len(self.context)

        if total_tokens <= 8000:
            # Small enough — dump full context into the prompt
            lines = []
            for i, msg in enumerate(self.context):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                lines.append(f"[{i}] {role}: {content}")
            return (
                f"CONVERSATION CONTEXT ({msg_count} messages, ~{total_tokens:,} tokens):\n\n"
                + "\n\n".join(lines)
            )
        else:
            # Too large for full inclusion — provide metadata and rely on SEARCH
            first_preview = self.context[0].get("content", "")[:200] if self.context else ""
            last_preview = self.context[-1].get("content", "")[:200] if self.context else ""
            return (
                f"CONVERSATION CONTEXT: {msg_count} messages, ~{total_tokens:,} tokens.\n"
                f"The context is too large to include in full. "
                f"Use SEARCH(query) to find relevant messages by keyword.\n\n"
                f"First message preview: {first_preview}...\n"
                f"Last message preview: {last_preview}...\n"
            )

    def run(self, user_input: str, max_turns: int = 15, add_user_msg: bool = True):
        if add_user_msg:
            self.add_to_context("user", user_input)

        # Build BM25 index over context
        self._build_bm25_index()

        # Persistent code execution environment
        env_locals: dict = {
            "print": print,
        }

        # Build the prompt — per Algorithm 2, the context goes INTO the LLM window
        context_block = self._build_context_summary()

        meta_prompt = (
            f"{context_block}\n\n"
            f"---\n\n"
            f"USER QUESTION: {user_input}\n\n"
            f"Respond to the question above. Use SEARCH() to find relevant information "
            f"in the context, use ```python code blocks to compute/verify, "
            f"then provide your final answer as ANSWER: [your answer]."
        )

        messages = [
            SystemMessage(content=REACT_SYSTEM_PROMPT),
            HumanMessage(content=meta_prompt),
        ]

        final_answer = None

        for turn in range(max_turns):
            try:
                response_msg = self.llm.invoke(messages)
                content = response_msg.content
            except Exception as e:
                yield {"type": "error", "content": str(e)}
                return

            yield {"type": "thought", "content": content}
            messages.append(AIMessage(content=content))

            observations = []

            # --- Execute SEARCH() calls ---
            search_matches = re.findall(r"SEARCH\(([^)]+)\)", content)
            for query in search_matches:
                query = query.strip().strip("\"'")
                results = self._search(query)
                search_output = f"SEARCH RESULTS for '{query}':\n\n{results}"
                observations.append(search_output)
                yield {"type": "observation", "content": search_output}

            # --- Execute code blocks ---
            code_blocks = re.findall(r"```(?:python)?(.*?)```", content, re.DOTALL)

            if code_blocks:
                for code in code_blocks:
                    code = textwrap.dedent(code).strip()
                    lines = code.splitlines()
                    if lines:
                        first = lines[0].strip()
                        if re.match(r'^python(?:\s|$)', first):
                            after_marker = re.sub(r'^python\s*', '', first)
                            if not after_marker or after_marker.startswith('#'):
                                code = "\n".join(lines[1:]).strip()
                            else:
                                rest = "\n".join(lines[1:])
                                code = (after_marker + "\n" + rest).strip()

                    executable = [
                        l for l in code.splitlines()
                        if l.strip() and not l.strip().startswith('#')
                    ]
                    if not executable:
                        continue

                    print(f"\n[REACT CODE]\n{code}\n[/REACT CODE]", flush=True)

                    f_out = io.StringIO()
                    f_err = io.StringIO()

                    with contextlib.redirect_stdout(f_out), contextlib.redirect_stderr(f_err):
                        try:
                            exec(code, env_locals)
                        except Exception as e:
                            print(f"Runtime Error: {e}", file=f_err)

                    stdout = f_out.getvalue()
                    stderr = f_err.getvalue()
                    full_output = (stdout + stderr).strip()

                    print(
                        f"[REACT OUTPUT]\n{full_output or '[No Output]'}\n[/REACT OUTPUT]",
                        flush=True,
                    )

                    if full_output:
                        if len(full_output) > 2000:
                            full_output = full_output[:2000] + "\n... [Output Truncated]"
                        display = f"[Code]\n{code}\n\n[Output]\n{full_output}"
                    else:
                        display = f"[Code]\n{code}\n\n[No Output]"

                    observations.append(display)
                    yield {"type": "observation", "content": display}

            # Feed observations back to the model
            if observations:
                combined = "\n\n---\n\n".join(observations)
                messages.append(HumanMessage(content=combined))

            # --- Check for ANSWER: ---
            parsed = self._parse_answer(content)
            if parsed is not None:
                final_answer = parsed
                yield {"type": "final", "content": final_answer}
                break

            # No action guard
            if not code_blocks and not search_matches and parsed is None:
                messages.append(HumanMessage(
                    content=(
                        "No action detected. Use SEARCH(query) to find information, "
                        "write ```python code to compute, or provide ANSWER: [your answer]."
                    )
                ))

        if not final_answer:
            force_prompt = (
                "You have used all reasoning turns. You MUST give a final answer RIGHT NOW. "
                "Do NOT write any more code or searches. "
                "Provide your answer as ANSWER: [your best answer here] based on everything gathered so far."
            )
            messages.append(HumanMessage(content=force_prompt))

            for _ in range(3):
                try:
                    response_msg = self.llm.invoke(messages)
                    content = response_msg.content
                except Exception as e:
                    yield {"type": "error", "content": str(e)}
                    break

                yield {"type": "thought", "content": content}
                messages.append(AIMessage(content=content))

                parsed = self._parse_answer(content)
                if parsed is not None:
                    final_answer = parsed
                    yield {"type": "final", "content": final_answer}
                    break

                messages.append(HumanMessage(
                    content="Still no ANSWER detected. You MUST use ANSWER: [your answer] now."
                ))

            if not final_answer:
                final_answer = "Unable to produce a final answer after maximum attempts."
                yield {"type": "final", "content": final_answer}

        self.add_to_context("assistant", final_answer)
