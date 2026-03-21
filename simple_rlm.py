import os
import re
import io
import time
import textwrap
import concurrent.futures
import contextlib
from typing import Any, List, Dict, Optional
from dotenv import load_dotenv

# New import for accurate token counting
import tiktoken

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# --- Configuration ---
MODEL_NAME = "nvidia/nemotron-3-super-120b-a12b"
TEMPERATURE = 0.9
TOP_P = 0.95

try:
    with open("rlm_system_prompt.txt", "r") as f:
        RLM_SYSTEM_PROMPT = f.read()
except FileNotFoundError:
    RLM_SYSTEM_PROMPT = "Error: rlm_system_prompt.txt not found."

def _parse_final(content: str, env_locals: dict):
    """
    Parse FINAL or FINAL_VAR from model output.
    Handles: FINAL(answer), FINAL: answer, FINAL_VAR(var_name)
    Returns (final_answer, match_type) or (None, None) if no match.
    """
    # 1. Check FINAL_VAR(variable_name)
    match_var = re.search(r"FINAL_VAR\((.*?)\)", content)
    if match_var:
        var_name = match_var.group(1).strip()
        if var_name in env_locals:
            return str(env_locals[var_name]), "var"
        return None, "var_missing"

    # 2. Check FINAL(answer) — parenthesized form
    match_paren = re.search(r"FINAL\(([\s\S]*?)\)", content)
    if match_paren:
        candidate = match_paren.group(1).strip()
        if candidate in env_locals and not candidate.startswith(('"', "'")):
            return str(env_locals[candidate]), "paren"
        # Strip surrounding quotes
        if len(candidate) >= 2 and candidate[0] in ('"', "'") and candidate[-1] == candidate[0]:
            candidate = candidate[1:-1]
        return candidate, "paren"

    # 3. Check FINAL: answer — colon form (common model mistake)
    match_colon = re.search(r"FINAL:\s*(.+)", content, re.DOTALL)
    if match_colon:
        candidate = match_colon.group(1).strip()
        # Trim trailing code fences or other artifacts
        candidate = re.split(r"\n```", candidate)[0].strip()
        if candidate:
            return candidate, "colon"

    return None, None


class RLM:
    def __init__(self, api_key: str = None, max_depth: int = 1):
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY not found.")

        self.llm = ChatNVIDIA(
            model=MODEL_NAME,
            nvidia_api_key=self.api_key,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )

        # Initialize Tokenizer (cl100k_base is standard for modern LLMs)
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # Fallback to gpt-4 encoding if specific name fails
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")

        self.max_depth = max_depth
        self.context: List[Dict[str, Any]] = []

    def _count_tokens(self, text: str) -> int:
        """
        Accurate token counting using Tiktoken.
        """
        if not text: 
            return 0
        try:
            return len(self.tokenizer.encode(str(text)))
        except Exception:
            # Fallback for edge cases
            return len(str(text)) // 2 

    def _llm_query(self, prompt: str, _retries: int = 5) -> str:
        """The recursive primitive with retry/backoff for rate limits."""
        for attempt in range(_retries):
            try:
                msg = self.llm.invoke([HumanMessage(content=prompt)])
                return msg.content.encode("utf-8", errors="replace").decode("utf-8")
            except Exception as e:
                err = str(e).lower()
                if ("429" in err or "rate" in err or "too many" in err) and attempt < _retries - 1:
                    wait = 2 ** attempt + 1
                    print(f"[RLM] Rate limited, retrying in {wait}s (attempt {attempt+1}/{_retries})...")
                    time.sleep(wait)
                    continue
                return f"Error in llm_query: {e}"

    def _llm_query_batched(self, prompts: List[str]) -> List[str]:
        """Concurrent execution of llm_query with throttled concurrency."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(self._llm_query, prompts))
        return results

    def add_to_context(self, role: str, content: str):
        """Adds a message to the persistent context state."""
        self.context.append({
            "role": role,
            "content": content,
            "tokens": self._count_tokens(content),
            "timestamp": time.time()
        })

    def run(self, user_input: str, max_turns: int = 15, add_user_msg: bool = True):
        """
        Executes the RLM loop for a single turn of conversation.
        """
        # 1. Update Context with User Input
        if add_user_msg:
            self.add_to_context("user", user_input)
        
        # 2. Setup Environment
        env_locals = {
            "context": self.context,
            "llm_query": self._llm_query,
            "llm_query_batched": self._llm_query_batched,
            "max_depth": self.max_depth,
            "print": print,
            "SHOW_VARS": lambda: list(env_locals.keys()),
            # No-ops so FINAL/FINAL_VAR inside a code block don't crash exec
            # (the regex parser handles them outside exec)
            "FINAL": lambda _: None,
            "FINAL_VAR": lambda _: None,
        }
        
        # 3. Build Controller Prompt
        total_tokens = sum(m["tokens"] for m in self.context)
        msg_count = len(self.context)
        
        meta_prompt = (
            f"USER MESSAGE: {user_input}\n\n"
            f"CONTEXT STATE: List[Dict] with {msg_count} messages, {total_tokens} total tokens.\n"
            f"SUB-LLM DEPTH LIMIT: {self.max_depth} (max recursive llm_query call stack depth).\n"
            f"ACTION: Respond to the user message above. "
            f"Use ```repl``` code blocks to inspect `context` if needed, then provide FINAL(your answer)."
        )
        
        messages = [
            SystemMessage(content=RLM_SYSTEM_PROMPT),
            HumanMessage(content=meta_prompt)
        ]

        final_answer = None

        for turn in range(max_turns):
            # --- 1. Invoke Root LLM ---
            try:
                response_msg = self.llm.invoke(messages)
                content = response_msg.content
            except Exception as e:
                yield {"type": "error", "content": str(e)}
                return

            yield {"type": "thought", "content": content}
            messages.append(AIMessage(content=content))

            # --- 2. Execute Code (Priority 1) ---
            code_blocks = re.findall(r"```(?:repl|python)?(.*?)```", content, re.DOTALL)
            block_outputs = []

            if code_blocks:
                for code in code_blocks:
                    code = textwrap.dedent(code).strip()
                    lines = code.splitlines()
                    if lines:
                        first = lines[0].strip()
                        if re.match(r'^(?:repl|python)(?:\s|$)', first):
                            after_marker = re.sub(r'^(?:repl|python)\s*', '', first)
                            if not after_marker or after_marker.startswith('#'):
                                # First line is just a marker/comment — drop it
                                code = "\n".join(lines[1:]).strip()
                            else:
                                # Code follows the marker on the same line — keep it
                                rest = "\n".join(lines[1:])
                                code = (after_marker + "\n" + rest).strip()

                    # If code is empty or only comments after cleanup, report back
                    executable = [l for l in code.splitlines() if l.strip() and not l.strip().startswith('#')]
                    if not executable:
                        err_msg = (
                            "SYSTEM ERROR: Your code block was rejected because all statements were on a SINGLE LINE "
                            "(no newline characters). Python requires each statement on its own line. "
                            "You MUST write your code with actual line breaks:\n"
                            "```repl\n"
                            "line1 = something\n"
                            "print(line1)\n"
                            "```\n"
                            "Try again NOW with proper newlines between every statement."
                        )
                        print(f"[RLM CODE SKIPPED - NO EXECUTABLE LINES]", flush=True)
                        block_outputs.append(err_msg)
                        yield {"type": "observation", "content": err_msg}
                        continue

                    print(f"\n[RLM CODE]\n{code}\n[/RLM CODE]", flush=True)

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

                    print(f"[RLM OUTPUT]\n{full_output or '[No Output]'}\n[/RLM OUTPUT]", flush=True)

                    if full_output:
                        if len(full_output) > 2000:
                            full_output = full_output[:2000] + "\n... [Output Truncated]"
                        display_output = f"[Code]\n{code}\n\n[Output]\n{full_output}"
                    else:
                        display_output = f"[Code]\n{code}\n\n[No Output]"

                    block_outputs.append(display_output)
                    yield {"type": "observation", "content": display_output}

                combined_observation = "\n".join([f"OUTPUT:\n{out}" for out in block_outputs])
                messages.append(HumanMessage(content=combined_observation))

            # --- 3. Check for FINAL / FINAL_VAR (Priority 2) ---
            parsed, match_type = _parse_final(content, env_locals)
            if match_type == "var_missing":
                err = f"System Error: Variable not found. Use SHOW_VARS() to check available variables."
                messages.append(HumanMessage(content=err))
                yield {"type": "observation", "content": err}
                continue
            elif parsed is not None:
                final_answer = parsed
                yield {"type": "final", "content": final_answer}
                break

            # --- 4. No Action Guard ---
            if not code_blocks and parsed is None:
                messages.append(HumanMessage(content="No code or FINAL detected. Please write code to read context."))

        if not final_answer:
            force_prompt = (
                "You have used all reasoning turns. You MUST give a final answer RIGHT NOW. "
                "Do NOT write any more code blocks. "
                "Use FINAL(your best answer here) based on everything gathered so far."
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

                parsed, match_type = _parse_final(content, env_locals)
                if parsed is not None:
                    final_answer = parsed
                    yield {"type": "final", "content": final_answer}
                    break

                messages.append(HumanMessage(content="Still no FINAL detected. You MUST use FINAL(your answer) now."))

            if not final_answer:
                final_answer = "Unable to produce a final answer after maximum attempts."
                yield {"type": "final", "content": final_answer}

        self.add_to_context("assistant", final_answer)