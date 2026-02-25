import os
import re
import io
import time
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
MODEL_NAME = "moonshotai/kimi-k2-instruct-0905"
TEMPERATURE = 0.9
MAX_TOKENS = 2048
TOP_P = 0.95

try:
    with open("rlm_system_prompt.txt", "r") as f:
        RLM_SYSTEM_PROMPT = f.read()
except FileNotFoundError:
    RLM_SYSTEM_PROMPT = "Error: rlm_system_prompt.txt not found."

class RLM:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY not found.")
            
        self.llm = ChatNVIDIA(
            model=MODEL_NAME,
            nvidia_api_key=self.api_key,
            temperature=TEMPERATURE,
            max_completion_tokens=MAX_TOKENS,
            top_p=TOP_P
        )
        
        # Initialize Tokenizer (cl100k_base is standard for modern LLMs)
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # Fallback to gpt-4 encoding if specific name fails
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")

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

    def _llm_query(self, prompt: str) -> str:
        """The recursive primitive."""
        try:
            msg = self.llm.invoke([HumanMessage(content=prompt)])
            return msg.content
        except Exception as e:
            return f"Error in llm_query: {e}"

    def _llm_query_batched(self, prompts: List[str]) -> List[str]:
        """Concurrent execution of llm_query."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
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

    def run(self, user_input: str, max_turns: int = 15):
        """
        Executes the RLM loop for a single turn of conversation.
        """
        # 1. Update Context with User Input
        self.add_to_context("user", user_input)
        
        # 2. Setup Environment
        env_locals = {
            "context": self.context,
            "llm_query": self._llm_query,
            "llm_query_batched": self._llm_query_batched,
            "print": print,
            "SHOW_VARS": lambda: list(env_locals.keys())
        }
        
        # 3. Build Controller Prompt
        total_tokens = sum(m["tokens"] for m in self.context)
        msg_count = len(self.context)
        
        meta_prompt = (
            f"STATUS: New user message received.\n"
            f"CONTEXT STATE: List[Dict] with {msg_count} messages.\n"
            f"TOTAL TOKENS: {total_tokens}.\n"
            f"ACTION: The latest message is at `context[-1]`. Write code to read it and generate a response."
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
            code_blocks = re.findall(r"```(?:repl|python)(.*?)```", content, re.DOTALL)
            block_outputs = []
            
            if code_blocks:
                for code in code_blocks:
                    f_out = io.StringIO()
                    f_err = io.StringIO()
                    
                    with contextlib.redirect_stdout(f_out), contextlib.redirect_stderr(f_err):
                        try:
                            exec(code, {}, env_locals)
                        except Exception as e:
                            print(f"Runtime Error: {e}", file=f_err)

                    stdout = f_out.getvalue()
                    stderr = f_err.getvalue()
                    
                    # Truncate strictly for observation prompt
                    full_output = stdout + stderr
                    if len(full_output) > 2000:
                        display_output = full_output[:2000] + f"\n... [Output Truncated]"
                    else:
                        display_output = full_output
                    
                    if not display_output.strip():
                        display_output = "[No Output]"

                    block_outputs.append(display_output)
                    yield {"type": "observation", "content": display_output}

                combined_observation = "\n".join([f"OUTPUT:\n{out}" for out in block_outputs])
                messages.append(HumanMessage(content=combined_observation))

            # --- 3. Check for FINAL / FINAL_VAR (Priority 2) ---
            
            # Check FINAL_VAR
            match_var = re.search(r"FINAL_VAR\((.*?)\)", content)
            if match_var:
                var_name = match_var.group(1).strip()
                if var_name in env_locals:
                    final_answer = str(env_locals[var_name])
                    yield {"type": "final", "content": final_answer}
                    break
                else:
                    err = f"System Error: Variable '{var_name}' does not exist."
                    messages.append(HumanMessage(content=err))
                    yield {"type": "observation", "content": err}
                    continue

            # Check FINAL
            match_text = re.search(r"FINAL\(([\s\S]*?)\)", content)
            if match_text:
                candidate = match_text.group(1).strip()
                # Smart fallback for quotes or variables
                if candidate in env_locals and not candidate.startswith(('"', "'")):
                     final_answer = str(env_locals[candidate])
                else:
                     final_answer = candidate
                
                yield {"type": "final", "content": final_answer}
                break

            # --- 4. No Action Guard ---
            if not code_blocks and not match_var and not match_text:
                 messages.append(HumanMessage(content="No code or FINAL detected. Please write code to read context."))

        if not final_answer:
            final_answer = "Max turns reached without resolution."
            yield {"type": "final", "content": final_answer}

        self.add_to_context("assistant", final_answer)