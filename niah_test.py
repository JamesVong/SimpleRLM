"""
Multi-Needle-in-a-Haystack (NIAH) Test for RLM & ReAct Agents

Simulates a long chat history where books serve as prior conversation context.
Multiple needle paragraphs are scattered at different depths throughout the
agent's context, then the agent is asked to retrieve and synthesize them.

Tests both the RLM and ReAct agent implementations side-by-side.
"""

import os
import re
import time
import json
import random
import urllib.request
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import tiktoken

from simple_rlm import RLM
from react_agent import ReActAgent

load_dotenv()

BOOKS = {
    "Pride and Prejudice": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
    "Frankenstein": "https://www.gutenberg.org/cache/epub/84/pg84.txt",
    "Sherlock Holmes": "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
    "Moby Dick": "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
    "War and Peace": "https://www.gutenberg.org/cache/epub/2600/pg2600.txt",
    "Great Expectations": "https://www.gutenberg.org/cache/epub/1400/pg1400.txt",
    "Les Miserables": "https://www.gutenberg.org/cache/epub/135/pg135.txt",
    "Anna Karenina": "https://www.gutenberg.org/cache/epub/1399/pg1399.txt",
}

# --- Multi-Needle Definitions ---
NEEDLES = [
    {
        "id": "project_alpha",
        "text": (
            "INTERNAL MEMO — Project Alpha: The satellite deployment window has been "
            "moved to March 15th. The launch coordinates are 28.5°N, 80.6°W. The mission "
            "commander is Dr. Elena Vasquez, and the backup frequency is 142.7 MHz. All "
            "teams must confirm readiness by March 10th."
        ),
        "question": "When is the Project Alpha satellite deployment, and what are the launch coordinates?",
        "expected_fragments": ["March 15", "28.5", "80.6"],
    },
    {
        "id": "budget_report",
        "text": (
            "CONFIDENTIAL BUDGET NOTE: The Q3 infrastructure budget was revised to "
            "$4.7 million, down from the original $6.2 million estimate. The reduction "
            "came from consolidating the Redis cluster (saving $800K) and renegotiating "
            "the CloudVault contract to $1.1 million per year. CFO Margaret Chen approved "
            "the revised figures on Tuesday."
        ),
        "question": "What is the revised Q3 infrastructure budget and how much was saved on the Redis cluster?",
        "expected_fragments": ["4.7 million", "800"],
    },
    {
        "id": "security_incident",
        "text": (
            "SECURITY INCIDENT #4491: On January 22nd at 03:14 UTC, an unauthorized access "
            "attempt was detected on the prod-db-west-2 database cluster. The attacker IP "
            "was traced to 198.51.100.47. The breach was contained within 12 minutes by the "
            "on-call engineer, Raj Patel. Root cause: an expired TLS certificate on the "
            "ingress controller allowed a downgrade attack."
        ),
        "question": "What was the attacker IP in security incident #4491 and what was the root cause?",
        "expected_fragments": ["198.51.100.47", "TLS certificate", "downgrade"],
    },
    {
        "id": "recipe_secret",
        "text": (
            "GRANDMOTHER'S SECRET RECIPE — The key to the perfect sourdough is exactly "
            "78% hydration, with 2 tablespoons of honey per 500g of flour. The starter "
            "must be fed at exactly 11pm the night before, and the dough must proof for "
            "precisely 13.5 hours at 72°F. Grandmother always said the magic ingredient "
            "was a pinch of cardamom."
        ),
        "question": "What hydration percentage and secret ingredient does grandmother's sourdough recipe use?",
        "expected_fragments": ["78%", "cardamom"],
    },
    {
        "id": "experiment_results",
        "text": (
            "LAB NOTEBOOK — Experiment #X-7742: The crystallization threshold was observed "
            "at exactly -41.3°C under 2.8 atmospheres of pressure. The sample (batch ID: "
            "NQ-8819) exhibited a previously undocumented phase transition lasting 0.73 "
            "seconds. Dr. Kimura noted that the lattice structure resembled hexagonal "
            "close-packed with an anomalous 7.2° tilt angle."
        ),
        "question": "What temperature and pressure triggered crystallization in experiment X-7742, and what was the tilt angle?",
        "expected_fragments": ["-41.3", "2.8 atmospheres", "7.2"],
    },
]

# --- Test Configurations ---
CONTEXT_LENGTHS = [2_000_000, 10_000_000]
NEEDLE_COUNTS = [5]
TOKENS_PER_TURN = 800


def download_text(url: str, retries: int = 3, timeout: int = 30) -> str:
    """Download a text file from a URL with retry logic."""
    for attempt in range(retries):
        try:
            print(f"  Downloading {url} (attempt {attempt + 1}/{retries}) ...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            start = re.search(r"\*\*\* START OF .+\*\*\*", raw)
            end = re.search(r"\*\*\* END OF .+\*\*\*", raw)
            if start and end:
                raw = raw[start.end():end.start()]
            return raw.strip()
        except Exception as e:
            print(f"    Failed: {e}")
            if attempt < retries - 1:
                wait = 5 * (attempt + 1)
                print(f"    Retrying in {wait}s...")
                time.sleep(wait)
    return ""


def load_haystack_texts(cache_dir: str = "niah_cache") -> str:
    """Download and concatenate all book texts into one big haystack."""
    os.makedirs(cache_dir, exist_ok=True)
    full_text = ""
    for title, url in BOOKS.items():
        cache_path = os.path.join(cache_dir, f"{title.replace(' ', '_')}.txt")
        if os.path.exists(cache_path):
            print(f"  Using cached: {title}")
            with open(cache_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            text = download_text(url)
            if not text:
                print(f"  SKIPPED: {title} (download failed)")
                continue
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(text)
        full_text += f"\n\n{text}"
    return full_text


def build_chat_context(
    haystack_text: str,
    tokenizer: tiktoken.Encoding,
    target_tokens: int,
    needles: List[Dict],
) -> Tuple[List[Dict], Dict[str, float]]:
    """
    Build a simulated chat context (List[Dict]) with needles injected.
    Each entry matches the agent's context format:
      {"role": str, "content": str, "tokens": int, "timestamp": float}
    """
    # Trim or repeat haystack to fill target
    haystack_tokens = tokenizer.encode(haystack_text)
    needle_overhead = sum(len(tokenizer.encode(n["text"])) for n in needles)
    available = target_tokens - needle_overhead - 500  # reserve for question

    if len(haystack_tokens) < available:
        repeats = (available // len(haystack_tokens)) + 1
        haystack_tokens = (haystack_tokens * repeats)[:available]
    else:
        haystack_tokens = haystack_tokens[:available]

    trimmed_text = tokenizer.decode(haystack_tokens)

    # Chunk into alternating user/assistant turns
    tokens = tokenizer.encode(trimmed_text)
    turns = []
    i = 0
    is_user = True
    base_time = time.time() - 86400  # start "yesterday"

    while i < len(tokens):
        chunk_tokens = tokens[i : i + TOKENS_PER_TURN]
        chunk_text = tokenizer.decode(chunk_tokens)

        if is_user:
            content = f"Here's the next passage I'd like to discuss:\n\n{chunk_text}"
            role = "user"
        else:
            content = (
                f"Thank you for sharing that passage. It covers interesting themes. "
                f"Here's a brief passage that continues the narrative:\n\n{chunk_text}"
            )
            role = "assistant"

        turns.append({
            "role": role,
            "content": content,
            "tokens": len(chunk_tokens) + 20,  # rough overhead for wrapper text
            "timestamp": base_time + len(turns) * 2,
        })
        i += TOKENS_PER_TURN
        is_user = not is_user

    # Inject needle turns at evenly spaced depths
    n = len(needles)
    total = len(turns)
    positions = [int(total * (i + 1) / (n + 1)) for i in range(n)]
    depth_map = {}

    for needle, pos in sorted(zip(needles, positions), key=lambda x: -x[1]):
        needle_turn = {
            "role": "user",
            "content": (
                f"Oh, before I forget — I came across this important note:\n\n"
                f"{needle['text']}\n\n"
                f"Anyway, let's continue with the reading."
            ),
            "tokens": len(tokenizer.encode(needle["text"])) + 30,
            "timestamp": base_time + pos * 2,
        }
        turns.insert(pos, needle_turn)
        depth_map[needle["id"]] = round(pos / total, 2)

    return turns, depth_map


def build_retrieval_question(needles: List[Dict]) -> str:
    """Build a combined question asking about all needles."""
    if len(needles) == 1:
        return needles[0]["question"]

    parts = [f"{i+1}. {n['question']}" for i, n in enumerate(needles)]
    return (
        "Based on our entire conversation history, please answer ALL of the "
        "following questions. Be specific and include exact numbers/details:\n\n"
        + "\n".join(parts)
    )


def score_response(response: str, needles: List[Dict]) -> Dict:
    """Score how many needle fragments were found in the response."""
    results = {}
    for needle in needles:
        found = []
        missing = []
        for frag in needle["expected_fragments"]:
            if frag.lower() in response.lower():
                found.append(frag)
            else:
                missing.append(frag)
        results[needle["id"]] = {
            "found": found,
            "missing": missing,
            "score": len(found) / len(needle["expected_fragments"]),
        }
    return results


def run_agent(agent, agent_name: str, question: str) -> Tuple[str, float, List[Dict]]:
    """
    Run an agent and collect its final answer + trace.
    Returns (final_answer, elapsed_seconds, trace).
    """
    trace = []
    final_answer = ""
    start = time.time()

    for event in agent.run(question, max_turns=10):
        trace.append(event)
        etype = event["type"]
        content = event["content"]

        if etype == "thought":
            preview = content[:300].replace('\n', ' ')
            print(f"    [{agent_name} THOUGHT] {preview}...")
        elif etype == "observation":
            preview = content[:300].replace('\n', ' ')
            print(f"    [{agent_name} OBSERVATION] {preview}...")
        elif etype == "final":
            final_answer = content
            preview = content[:200].replace('\n', ' ')
            print(f"    [{agent_name} FINAL] {preview}...")
        elif etype == "error":
            final_answer = f"ERROR: {content}"
            print(f"    [{agent_name} ERROR] {content[:300]}")

    elapsed = time.time() - start
    return final_answer, elapsed, trace


def run_test(
    agent_class,
    agent_name: str,
    tokenizer: tiktoken.Encoding,
    haystack_text: str,
    context_length: int,
    needle_count: int,
    agent_kwargs: dict = None,
) -> Dict:
    """Run a single NIAH test with an agent."""
    selected_needles = NEEDLES[:needle_count]

    # Build the chat context with needles
    context_turns, depth_map = build_chat_context(
        haystack_text, tokenizer, context_length, selected_needles
    )

    # Create agent and pre-load its context
    agent = agent_class(**(agent_kwargs or {}))
    agent.context = context_turns

    actual_tokens = sum(t["tokens"] for t in context_turns)
    print(f"    Chat turns: {len(context_turns)}, ~{actual_tokens:,} tokens")
    print(f"    Needle depths: {depth_map}")

    # Ask the retrieval question
    question = build_retrieval_question(selected_needles)
    answer, elapsed, trace = run_agent(agent, agent_name, question)

    # Score
    scores = score_response(answer, selected_needles)
    avg_score = sum(s["score"] for s in scores.values()) / len(scores)

    # Build trace summary for logging
    trace_summary = []
    for t in trace:
        trace_summary.append({
            "type": t["type"],
            "content": t["content"][:500],
        })

    return {
        "agent": agent_name,
        "context_length": context_length,
        "actual_tokens": actual_tokens,
        "needle_count": needle_count,
        "needle_depths": depth_map,
        "scores": scores,
        "avg_score": round(avg_score, 3),
        "answer": answer[:500],
        "time_s": round(elapsed, 2),
        "turns_used": len([t for t in trace if t["type"] == "thought"]),
        "trace": trace_summary,
    }


def main():
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print("ERROR: Set NVIDIA_API_KEY in your .env file")
        return

    tokenizer = tiktoken.get_encoding("cl100k_base")

    print("=" * 65)
    print("  Multi-Needle NIAH — RLM vs ReAct Agent Comparison")
    print("=" * 65)

    print("\nDownloading haystack texts...")
    haystack_text = load_haystack_texts()
    total_tokens = len(tokenizer.encode(haystack_text))
    print(f"Total haystack available: {total_tokens:,} tokens")
    if total_tokens < max(CONTEXT_LENGTHS):
        print(f"  (text will be repeated to fill larger context windows)")
    print()

    print(f"Needles defined: {len(NEEDLES)}")
    for n in NEEDLES:
        print(f"  - {n['id']}: {n['text'][:60]}...")
    print()

    agents = [
        (RLM, "RLM", {"max_depth": 1}),
        (ReActAgent, "ReAct", {}),
    ]

    results = []

    for ctx_len in CONTEXT_LENGTHS:
        for n_count in NEEDLE_COUNTS:
            for agent_class, agent_name, agent_kwargs in agents:
                label = f"{agent_name:>5} | ctx={ctx_len:>7,} | needles={n_count}"
                print(f"\n{'='*55}")
                print(f"  {label}")
                print(f"{'='*55}")

                try:
                    result = run_test(
                        agent_class, agent_name, tokenizer,
                        haystack_text, ctx_len, n_count,
                        agent_kwargs=agent_kwargs,
                    )
                except Exception as e:
                    print(f"  FAILED: {e}")
                    results.append({
                        "agent": agent_name,
                        "context_length": ctx_len,
                        "needle_count": n_count,
                        "avg_score": 0,
                        "error": str(e),
                    })
                    continue

                # Print per-needle scores
                for nid, score_info in result["scores"].items():
                    status = "PASS" if score_info["score"] == 1.0 else (
                        "PARTIAL" if score_info["score"] > 0 else "FAIL"
                    )
                    print(f"  {nid:>20}: {status} ({score_info['score']:.0%}) "
                          f"found={score_info['found']} missing={score_info['missing']}")

                print(f"  {'AVERAGE':>20}: {result['avg_score']:.0%}  "
                      f"({result['time_s']}s, {result['turns_used']} reasoning turns)")
                print(f"  Answer: {result['answer'][:150]}...")
                results.append(result)

    # === Summary ===
    print("\n" + "=" * 65)
    print("  SUMMARY — RLM vs ReAct")
    print("=" * 65)

    for agent_name in ["RLM", "ReAct"]:
        print(f"\n  {agent_name}:")
        header = f"  {'Context':>10} |"
        for nc in NEEDLE_COUNTS:
            header += f" {nc} needle{'s' if nc > 1 else ' ':1} |"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for ctx_len in CONTEXT_LENGTHS:
            row = f"  {ctx_len:>10,} |"
            for nc in NEEDLE_COUNTS:
                match = [r for r in results
                         if r.get("agent") == agent_name
                         and r["context_length"] == ctx_len
                         and r["needle_count"] == nc]
                if match and "avg_score" in match[0]:
                    score = match[0]["avg_score"]
                    row += f"  {score:>5.0%}  |"
                else:
                    row += f"  ERROR |"
            print(row)

    # Save full results
    output_path = "niah_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
