import streamlit as st
import os
import time
import random
import threading
import queue
import tiktoken
from simple_rlm import RLM
from react_agent import ReActAgent
from hard_needles import TEST_SUITES, score_response_hard
import db

# --- Page Config ---
st.set_page_config(page_title="RLM vs ReAct", layout="wide")

TOKENIZER = tiktoken.get_encoding("cl100k_base")
NEEDLE_CACHE_DIR = "niah_cache"
TOKENS_PER_TURN = 800


# --- Initialize Agents ---
def _init_agents(api_key):
    st.session_state.rlm_instance = RLM(api_key=api_key, max_depth=1)
    st.session_state.react_instance = ReActAgent(api_key=api_key)


if "rlm_instance" not in st.session_state:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if api_key:
        try:
            _init_agents(api_key)
        except Exception:
            st.session_state.rlm_instance = None
            st.session_state.react_instance = None
    else:
        st.session_state.rlm_instance = None
        st.session_state.react_instance = None

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

# Needle lab state
if "needle_texts" not in st.session_state:
    st.session_state.needle_texts = [""] * 5
if "needle_count" not in st.session_state:
    st.session_state.needle_count = 3
if "needle_haystack" not in st.session_state:
    st.session_state.needle_haystack = None   # built context turns (List[Dict])
if "needle_haystack_source" not in st.session_state:
    st.session_state.needle_haystack_source = "books"  # "books" or "chat"


# --- Helpers ---
def switch_chat(chat_id):
    st.session_state.current_chat_id = chat_id
    if st.session_state.rlm_instance:
        st.session_state.rlm_instance.context = db.load_messages(chat_id, models=["user", "rlm"])
    if st.session_state.react_instance:
        st.session_state.react_instance.context = db.load_messages(chat_id, models=["user", "react"])


def create_new_chat():
    new_id = db.create_conversation("New Chat")
    st.session_state.current_chat_id = new_id
    if st.session_state.rlm_instance:
        st.session_state.rlm_instance.context = []
    if st.session_state.react_instance:
        st.session_state.react_instance.context = []


def delete_current_chat():
    if st.session_state.current_chat_id:
        db.delete_conversation(st.session_state.current_chat_id)
        st.session_state.current_chat_id = None
        if st.session_state.rlm_instance:
            st.session_state.rlm_instance.context = []
        if st.session_state.react_instance:
            st.session_state.react_instance.context = []
        st.rerun()


def render_sidebar_stats(rlm_msgs, react_msgs):
    with st.session_state.stats_placeholder.container():
        st.subheader("Context State")
        col1, col2 = st.columns(2)
        with col1:
            st.caption("RLM")
            st.metric("Messages", len(rlm_msgs))
            st.metric("Tokens", sum(m.get("tokens", 0) for m in rlm_msgs))
        with col2:
            st.caption("ReAct")
            st.metric("Messages", len(react_msgs))
            st.metric("Tokens", sum(m.get("tokens", 0) for m in react_msgs))
        st.divider()


def _load_books_text() -> str:
    """Load all cached book texts concatenated."""
    if not os.path.isdir(NEEDLE_CACHE_DIR):
        return ""
    parts = []
    for fname in sorted(os.listdir(NEEDLE_CACHE_DIR)):
        if fname.endswith(".txt"):
            with open(os.path.join(NEEDLE_CACHE_DIR, fname), "r", encoding="utf-8") as f:
                parts.append(f.read())
    return "\n\n".join(parts)


def _build_haystack_from_text(raw_text: str, target_tokens: int, needle_texts: list) -> list:
    """
    Chunk raw_text into alternating user/assistant turns (~TOKENS_PER_TURN each),
    then inject needle_texts at evenly-spaced positions.
    Returns List[Dict] matching agent context format.
    """
    haystack_tokens = TOKENIZER.encode(raw_text)
    needle_overhead = sum(len(TOKENIZER.encode(n)) for n in needle_texts if n.strip())
    available = max(target_tokens - needle_overhead - 500, TOKENS_PER_TURN)

    if len(haystack_tokens) < available:
        repeats = (available // len(haystack_tokens)) + 1
        haystack_tokens = (haystack_tokens * repeats)[:available]
    else:
        haystack_tokens = haystack_tokens[:available]

    trimmed = TOKENIZER.decode(haystack_tokens)
    tokens = TOKENIZER.encode(trimmed)

    turns = []
    base_time = time.time() - 86400
    is_user = True
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i: i + TOKENS_PER_TURN]
        chunk_text = TOKENIZER.decode(chunk_tokens)
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
            "tokens": len(chunk_tokens) + 20,
            "timestamp": base_time + len(turns) * 2,
        })
        i += TOKENS_PER_TURN
        is_user = not is_user

    # Inject needles at evenly-spaced positions
    active_needles = [n for n in needle_texts if n.strip()]
    n = len(active_needles)
    if n and len(turns) > 0:
        total = len(turns)
        positions = [int(total * (idx + 1) / (n + 1)) for idx in range(n)]
        random.shuffle(positions)
        for needle_text, pos in sorted(zip(active_needles, positions), key=lambda x: -x[1]):
            needle_turn = {
                "role": "user",
                "content": (
                    f"Oh, before I forget — I came across this important note:\n\n"
                    f"{needle_text}\n\n"
                    f"Anyway, let's continue with the reading."
                ),
                "tokens": len(TOKENIZER.encode(needle_text)) + 30,
                "timestamp": base_time + pos * 2,
            }
            turns.insert(pos, needle_turn)

    return turns


def _count_context_tokens(turns: list) -> int:
    return sum(t.get("tokens", 0) for t in turns)


# --- Sidebar ---
with st.sidebar:
    st.title("RLM vs ReAct")

    if not st.session_state.rlm_instance:
        api_key = st.text_input("NVIDIA API Key", type="password")
        if api_key:
            os.environ["NVIDIA_API_KEY"] = api_key
            try:
                _init_agents(api_key)
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()
    st.session_state.stats_placeholder = st.empty()

    if st.button("➕ New Conversation", use_container_width=True):
        create_new_chat()
        st.rerun()

    st.markdown("### History")
    chats = db.get_all_conversations()
    for chat in chats:
        label = chat["title"]
        if st.session_state.current_chat_id == chat["id"]:
            label = f"🟢 {label}"
        if st.button(label, key=f"chat_{chat['id']}", use_container_width=True):
            switch_chat(chat["id"])
            st.rerun()

    if st.session_state.current_chat_id:
        st.divider()
        if st.button("🗑️ Delete Current Chat", type="primary"):
            delete_current_chat()


# --- Ensure a chat is selected ---
if st.session_state.current_chat_id is None:
    if not chats:
        create_new_chat()
        st.rerun()
    else:
        st.title("RLM vs ReAct Comparison")
        st.info("Select a conversation from the sidebar or start a new one.")
        st.stop()

chat_id = st.session_state.current_chat_id
chat_title = next((c["title"] for c in chats if c["id"] == chat_id), "New Chat")

# --- Load message histories ---
rlm_msgs = db.load_messages(chat_id, models=["user", "rlm"])
react_msgs = db.load_messages(chat_id, models=["user", "react"])

# --- Sync agent contexts & stats ---
if st.session_state.rlm_instance:
    st.session_state.rlm_instance.context = rlm_msgs
if st.session_state.react_instance:
    st.session_state.react_instance.context = react_msgs
render_sidebar_stats(rlm_msgs, react_msgs)

# --- Header ---
st.header(chat_title)

# --- Tabs ---
tab_chat, tab_needle = st.tabs(["💬 Chat", "🪡 Needle Lab"])


# =========================================================
#  TAB 1 — Chat
# =========================================================
with tab_chat:
    col_rlm, col_react = st.columns(2)

    with col_rlm:
        st.subheader("RLM — Recursive")
        for msg in rlm_msgs:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    with col_react:
        st.subheader("ReAct — Code Only")
        for msg in react_msgs:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt := st.chat_input("Type your message...", key="chat_input"):
        if not st.session_state.rlm_instance:
            st.error("Please enter an API Key in the sidebar.")
            st.stop()

        db.save_message(chat_id, "user", prompt, tokens=len(prompt) // 4, model="user")

        with col_rlm:
            with st.chat_message("user"):
                st.markdown(prompt)
        with col_react:
            with st.chat_message("user"):
                st.markdown(prompt)

        rlm_agent = st.session_state.rlm_instance
        react_inst = st.session_state.react_instance

        rlm_q = queue.Queue()
        react_q = queue.Queue()

        def _run_rlm(agent, q):
            try:
                agent.context = db.load_messages(chat_id, models=["user", "rlm"])
                for step in agent.run(prompt, add_user_msg=False):
                    q.put(step)
            finally:
                q.put(None)

        def _run_react(agent, q):
            try:
                agent.context = db.load_messages(chat_id, models=["user", "react"])
                for step in agent.run(prompt, add_user_msg=False):
                    q.put(step)
            finally:
                q.put(None)

        with col_rlm:
            rlm_chat = st.chat_message("assistant")
        with col_react:
            react_chat = st.chat_message("assistant")

        with rlm_chat:
            rlm_status = st.status("RLM thinking recursively...", expanded=True)
            rlm_answer = st.empty()
        with react_chat:
            react_status = st.status("ReAct reasoning...", expanded=True)
            react_answer = st.empty()

        t1 = threading.Thread(target=_run_rlm, args=(rlm_agent, rlm_q))
        t2 = threading.Thread(target=_run_react, args=(react_inst, react_q))
        t1.start()
        t2.start()

        rlm_done = react_done = False
        rlm_final = react_final = ""

        while not (rlm_done and react_done):
            for q, done_flag, status, answer_ph, thought_label in [
                (rlm_q,   rlm_done,   rlm_status,   rlm_answer,   "**Thinking**"),
                (react_q, react_done, react_status, react_answer, "**Reasoning**"),
            ]:
                if done_flag:
                    continue
                try:
                    while True:
                        step = q.get_nowait()
                        if step is None:
                            if q is rlm_q:
                                rlm_done = True
                            else:
                                react_done = True
                            break
                        stype = step.get("type")
                        scontent = step.get("content", "")
                        if stype == "thought":
                            status.markdown(thought_label)
                            status.markdown(scontent)
                            status.markdown("---")
                        elif stype == "observation":
                            status.markdown("**Code Output**")
                            status.code(scontent)
                            status.markdown("---")
                        elif stype == "final":
                            if q is rlm_q:
                                rlm_final = scontent
                            else:
                                react_final = scontent
                            status.update(label="Done", state="complete", expanded=True)
                            answer_ph.markdown(scontent)
                        elif stype == "error":
                            status.error(scontent)
                            status.update(state="error", expanded=True)
                except queue.Empty:
                    pass

            if not (rlm_done and react_done):
                time.sleep(0.05)

        t1.join()
        t2.join()

        if rlm_final:
            db.save_message(chat_id, "assistant", rlm_final,
                            tokens=len(rlm_final) // 4, model="rlm")
        if react_final:
            db.save_message(chat_id, "assistant", react_final,
                            tokens=len(react_final) // 4, model="react")

        rlm_msgs = db.load_messages(chat_id, models=["user", "rlm"])
        react_msgs = db.load_messages(chat_id, models=["user", "react"])
        render_sidebar_stats(rlm_msgs, react_msgs)


# =========================================================
#  TAB 2 — Needle Lab
# =========================================================
with tab_needle:
    st.subheader("Needle Lab")
    st.caption(
        "Inject hidden facts into a large haystack, then ask both agents to find them. "
        "Needles are placed at random evenly-spaced positions."
    )

    # --- Hard Test Presets ---
    st.markdown("#### 🧪 Hard Test Presets")
    st.caption(
        "Load a research-grade test suite with pre-built adversarial needles, "
        "multi-hop reasoning, aggregation tasks, or contradiction detection."
    )
    preset_cols = st.columns(len(TEST_SUITES))
    for idx, (suite_key, suite_info) in enumerate(TEST_SUITES.items()):
        with preset_cols[idx]:
            if st.button(
                f"{suite_info['name']}",
                key=f"preset_{suite_key}",
                use_container_width=True,
                help=f"{suite_info['description']}\nComplexity: {suite_info['complexity']}",
            ):
                # Load suite needles into the needle text slots
                suite_needles = suite_info["needles"]
                new_texts = [""] * 5
                for i, needle in enumerate(suite_needles[:5]):
                    new_texts[i] = needle["text"]
                st.session_state.needle_texts = new_texts
                st.session_state.needle_count = min(len(suite_needles), 5)
                # Build the combined question
                parts = [f"{i+1}. {q['question']}" for i, q in enumerate(suite_info["questions"])]
                st.session_state.hard_preset_question = "\n".join(parts)
                st.session_state.hard_preset_suite = suite_key
                # For suites with >5 needles, store the extras
                if len(suite_needles) > 5:
                    st.session_state.needle_extras = suite_needles[5:]
                else:
                    st.session_state.needle_extras = []
                st.rerun()

    # Show active preset info
    if st.session_state.get("hard_preset_suite"):
        active_suite = TEST_SUITES.get(st.session_state.hard_preset_suite, {})
        n_needles = len(active_suite.get("needles", []))
        n_shown = min(n_needles, 5)
        extra = n_needles - n_shown
        preset_msg = (
            f"**Active preset:** {active_suite.get('name', '?')} — "
            f"{n_needles} needles, {len(active_suite.get('questions', []))} questions. "
            f"Complexity: {active_suite.get('complexity', '?')}"
        )
        if extra > 0:
            preset_msg += f"\n\n⚠️ {extra} additional needles beyond the 5 shown will also be injected."
        st.info(preset_msg)

    st.divider()

    # --- Step 1: Haystack source ---
    st.markdown("#### 1. Haystack source")
    source_col, size_col = st.columns([2, 1])
    with source_col:
        source = st.radio(
            "Build haystack from",
            ["Books (cached)", "Current chat history"],
            horizontal=True,
            key="needle_source_radio",
        )
    with size_col:
        target_tokens = st.number_input(
            "Target tokens",
            min_value=10_000,
            max_value=10_000_000,
            value=250_000,
            step=50_000,
            key="needle_target_tokens",
        )

    # --- Step 2: Define needles ---
    st.markdown("#### 2. Define needles")
    n_count = st.select_slider(
        "Number of needles",
        options=[1, 3, 5],
        value=st.session_state.needle_count,
        key="needle_count_slider",
    )
    st.session_state.needle_count = n_count

    needle_inputs = []
    cols = st.columns(min(n_count, 3))
    for i in range(n_count):
        col = cols[i % len(cols)]
        with col:
            val = st.text_area(
                f"Needle {i + 1}",
                value=st.session_state.needle_texts[i],
                height=110,
                placeholder="Type the hidden fact to inject...",
                key=f"needle_text_{i}",
            )
            needle_inputs.append(val)
            st.session_state.needle_texts[i] = val

    # --- Step 3: Build & preview haystack ---
    st.markdown("#### 3. Build haystack")
    build_col, _ = st.columns([1, 3])
    with build_col:
        build_btn = st.button("⚙️ Build Haystack", use_container_width=True)

    if build_btn:
        active_needles = [n for n in needle_inputs if n.strip()]
        # Also include extra needles from hard preset suites (>5 needles)
        extras = st.session_state.get("needle_extras", [])
        for extra_needle in extras:
            active_needles.append(extra_needle["text"])
        if not active_needles:
            st.warning("Enter at least one needle before building.")
        else:
            with st.spinner("Building haystack..."):
                if source == "Books (cached)":
                    raw = _load_books_text()
                    if not raw:
                        st.error(
                            f"No cached books found in `{NEEDLE_CACHE_DIR}/`. "
                            "Run the NIAH test once to download them."
                        )
                        st.stop()
                else:
                    # Use current chat context as raw text
                    all_msgs = db.load_messages(chat_id)
                    raw = "\n\n".join(m["content"] for m in all_msgs)
                    if not raw.strip():
                        st.warning("Current chat is empty. Switch to Books or add some messages.")
                        st.stop()

                turns = _build_haystack_from_text(raw, int(target_tokens), active_needles)
                st.session_state.needle_haystack = turns
                st.session_state.needle_haystack_source = source
            st.success(f"Haystack built: {len(turns):,} turns, ~{_count_context_tokens(turns):,} tokens")

    # --- Preview ---
    if st.session_state.needle_haystack:
        turns = st.session_state.needle_haystack
        total_tok = _count_context_tokens(turns)
        needle_turn_indices = [
            i for i, t in enumerate(turns)
            if "Oh, before I forget" in t["content"]
        ]

        st.markdown("#### Haystack preview")
        info_c, tok_c = st.columns([3, 1])
        with info_c:
            st.info(
                f"**{len(turns):,} turns** · **~{total_tok:,} tokens**  \n"
                f"Needle positions (turn index): {needle_turn_indices}"
            )
        with tok_c:
            depth_pcts = [f"{round(idx / len(turns) * 100)}%" for idx in needle_turn_indices]
            st.metric("Needle depths", ", ".join(depth_pcts) if depth_pcts else "—")

        # --- Needle quick-jump buttons ---
        if needle_turn_indices:
            st.markdown("**Jump to needle:**")
            jump_cols = st.columns(min(len(needle_turn_indices), 5))
            for j, nidx in enumerate(needle_turn_indices):
                with jump_cols[j % len(jump_cols)]:
                    if st.button(f"Needle {j+1} (turn {nidx})", key=f"jump_needle_{j}"):
                        # Set page so that the needle turn is visible
                        st.session_state.needle_preview_page = nidx // 50

        # --- Paginated preview ---
        PAGE_SIZE = 50
        total_pages = max(1, (len(turns) + PAGE_SIZE - 1) // PAGE_SIZE)
        if "needle_preview_page" not in st.session_state:
            st.session_state.needle_preview_page = 0
        current_page = st.session_state.needle_preview_page
        current_page = max(0, min(current_page, total_pages - 1))

        nav_left, nav_info, nav_right = st.columns([1, 2, 1])
        with nav_left:
            if st.button("◀ Prev", disabled=current_page == 0, key="prev_page"):
                st.session_state.needle_preview_page = current_page - 1
                st.rerun()
        with nav_info:
            st.markdown(f"**Page {current_page + 1} / {total_pages}** (turns {current_page * PAGE_SIZE}–{min((current_page + 1) * PAGE_SIZE, len(turns)) - 1})")
        with nav_right:
            if st.button("Next ▶", disabled=current_page >= total_pages - 1, key="next_page"):
                st.session_state.needle_preview_page = current_page + 1
                st.rerun()

        with st.expander("Browse turns", expanded=False):
            start = current_page * PAGE_SIZE
            end = min(start + PAGE_SIZE, len(turns))
            for i in range(start, end):
                t = turns[i]
                is_needle = "Oh, before I forget" in t["content"]
                badge = "🪡 NEEDLE" if is_needle else t["role"].upper()
                if is_needle:
                    color = "#b8860b"    # dark goldenrod
                    text_color = "#fff"
                elif t["role"] == "user":
                    color = "#2a3a4a"    # muted dark blue-grey
                    text_color = "#cdd"
                else:
                    color = "#2a3f2a"    # muted dark green
                    text_color = "#cdc"
                st.markdown(
                    f"<div style='background:{color};color:{text_color};border-radius:6px;"
                    f"padding:8px 12px;margin-bottom:6px'>"
                    f"<b>[{i}] {badge}</b> · {t['tokens']} tokens<br>"
                    f"<span style='font-size:0.85em'>{t['content'][:300].replace(chr(10), ' ')}...</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # --- Step 4: Ask question ---
        st.markdown("#### 4. Ask the retrieval question")
        default_q = st.session_state.get("hard_preset_question", "")
        question = st.text_area(
            "Question to ask both agents",
            value=default_q,
            placeholder="e.g. What are all the hidden facts mentioned in our conversation?",
            height=120,
            key="needle_question",
        )

        ask_col, _ = st.columns([1, 3])
        with ask_col:
            ask_btn = st.button(
                "🚀 Inject & Ask Both Agents",
                use_container_width=True,
                disabled=not question.strip(),
            )

        if ask_btn and question.strip():
            if not st.session_state.rlm_instance:
                st.error("Please enter an API Key in the sidebar.")
                st.stop()

            rlm_agent = st.session_state.rlm_instance
            react_inst = st.session_state.react_instance

            # Load haystack into agents (replacing their context)
            rlm_agent.context = list(turns)
            react_inst.context = list(turns)

            rlm_q2 = queue.Queue()
            react_q2 = queue.Queue()

            def _run_rlm_needle(agent, q):
                try:
                    for step in agent.run(question, add_user_msg=True):
                        q.put(step)
                finally:
                    q.put(None)

            def _run_react_needle(agent, q):
                try:
                    for step in agent.run(question, add_user_msg=True):
                        q.put(step)
                finally:
                    q.put(None)

            st.markdown("---")
            res_col_rlm, res_col_react = st.columns(2)

            with res_col_rlm:
                st.markdown("**RLM — Recursive**")
                rlm_status2 = st.status("RLM thinking...", expanded=True)
                rlm_answer2 = st.empty()
            with res_col_react:
                st.markdown("**ReAct — Code Only**")
                react_status2 = st.status("ReAct reasoning...", expanded=True)
                react_answer2 = st.empty()

            t1 = threading.Thread(target=_run_rlm_needle, args=(rlm_agent, rlm_q2))
            t2 = threading.Thread(target=_run_react_needle, args=(react_inst, react_q2))
            t1.start()
            t2.start()

            rlm_done2 = react_done2 = False
            rlm_final2 = react_final2 = ""

            while not (rlm_done2 and react_done2):
                for q, done_flag, status, answer_ph, label in [
                    (rlm_q2,   rlm_done2,   rlm_status2,   rlm_answer2,   "**Thinking**"),
                    (react_q2, react_done2, react_status2, react_answer2, "**Reasoning**"),
                ]:
                    if done_flag:
                        continue
                    try:
                        while True:
                            step = q.get_nowait()
                            if step is None:
                                if q is rlm_q2:
                                    rlm_done2 = True
                                else:
                                    react_done2 = True
                                break
                            stype = step.get("type")
                            scontent = step.get("content", "")
                            if stype == "thought":
                                status.markdown(label)
                                status.markdown(scontent)
                                status.markdown("---")
                            elif stype == "observation":
                                status.markdown("**Code Output**")
                                status.code(scontent)
                                status.markdown("---")
                            elif stype == "final":
                                if q is rlm_q2:
                                    rlm_final2 = scontent
                                else:
                                    react_final2 = scontent
                                status.update(label="Done", state="complete", expanded=True)
                                answer_ph.markdown(scontent)
                            elif stype == "error":
                                status.error(scontent)
                                status.update(state="error", expanded=True)
                    except queue.Empty:
                        pass

                if not (rlm_done2 and react_done2):
                    time.sleep(0.05)

            t1.join()
            t2.join()

            # --- Auto-score if hard preset is active ---
            active_preset = st.session_state.get("hard_preset_suite")
            if active_preset and active_preset in TEST_SUITES:
                suite_questions = TEST_SUITES[active_preset]["questions"]
                st.markdown("---")
                st.markdown("#### Automated Scoring")

                score_cols = st.columns(2)
                for col, label, answer in [
                    (score_cols[0], "RLM", rlm_final2),
                    (score_cols[1], "ReAct", react_final2),
                ]:
                    with col:
                        st.markdown(f"**{label}**")
                        if not answer:
                            st.warning("No answer produced.")
                            continue
                        scores = score_response_hard(answer, suite_questions)
                        total_score = 0
                        for qid, info in scores.items():
                            status_icon = "✅" if info["score"] >= 1.0 else (
                                "🟡" if info["score"] > 0 else "❌"
                            )
                            st.markdown(f"{status_icon} **{qid}**: {info['score']:.0%}")
                            if info["found"]:
                                st.caption(f"  Found: {', '.join(info['found'])}")
                            if info["missing"]:
                                st.caption(f"  Missing: {', '.join(info['missing'])}")
                            if info.get("poison_found"):
                                st.caption(f"  ⚠️ Poison detected: {', '.join(info['poison_found'])}")
                            total_score += info["score"]
                        avg = total_score / len(scores) if scores else 0
                        st.metric("Suite Average", f"{avg:.0%}")

            # Optionally save to chat history
            if rlm_final2 or react_final2:
                st.markdown("---")
                save_col, _ = st.columns([1, 3])
                with save_col:
                    if st.button("💾 Save results to chat history"):
                        db.save_message(chat_id, "user", f"[Needle Lab] {question}",
                                        tokens=len(question) // 4, model="user")
                        if rlm_final2:
                            db.save_message(chat_id, "assistant", rlm_final2,
                                            tokens=len(rlm_final2) // 4, model="rlm")
                        if react_final2:
                            db.save_message(chat_id, "assistant", react_final2,
                                            tokens=len(react_final2) // 4, model="react")
                        st.success("Saved to chat history.")
