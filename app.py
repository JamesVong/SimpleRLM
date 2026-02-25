import streamlit as st
import os
import time
from simple_rlm import RLM
import db  # Import our database module

# --- Page Config ---
st.set_page_config(page_title="RLM Chat History", layout="wide")

# --- Initialize RLM (Singleton logic) ---
if "rlm_instance" not in st.session_state:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if api_key:
        try:
            st.session_state.rlm_instance = RLM(api_key=api_key)
        except:
            st.session_state.rlm_instance = None
    else:
        st.session_state.rlm_instance = None

# --- Session State Management ---
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

# --- Helper Functions ---
def switch_chat(chat_id):
    st.session_state.current_chat_id = chat_id
    if st.session_state.rlm_instance:
        messages = db.load_messages(chat_id)
        st.session_state.rlm_instance.context = messages

def create_new_chat():
    new_id = db.create_conversation("New Chat")
    st.session_state.current_chat_id = new_id
    if st.session_state.rlm_instance:
        st.session_state.rlm_instance.context = []

def delete_current_chat():
    if st.session_state.current_chat_id:
        db.delete_conversation(st.session_state.current_chat_id)
        st.session_state.current_chat_id = None
        if st.session_state.rlm_instance:
            st.session_state.rlm_instance.context = []
        st.rerun()

def render_sidebar_stats(context_messages):
    """Renders the stats about the current loaded context."""
    if "stats_placeholder" not in st.session_state:
        st.session_state.stats_placeholder = st.sidebar.empty()
        
    with st.session_state.stats_placeholder.container():
        st.subheader("Hidden Context State")
        
        if not context_messages:
            st.caption("No active context.")
            return

        # Calculate metrics
        total_tok = sum(m.get('tokens', 0) for m in context_messages)
        msg_count = len(context_messages)
        
        # Display metrics
        c1, c2 = st.columns(2)
        c1.metric("Messages", msg_count)
        c2.metric("Est. Tokens", total_tok)
        
        with st.expander("View Raw Context"):
            st.json(context_messages)
        st.divider()

# --- Sidebar: History & Config ---
with st.sidebar:
    st.title("RLM Chat 🧠")
    
    # API Key Input
    if not st.session_state.rlm_instance:
        api_key = st.text_input("NVIDIA API Key", type="password")
        if api_key:
            os.environ["NVIDIA_API_KEY"] = api_key
            try:
                st.session_state.rlm_instance = RLM(api_key=api_key)
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    
    st.divider()
    
    # 1. RENDER STATS PLACEHOLDER (Will be filled later)
    st.session_state.stats_placeholder = st.empty()

    # 2. Controls
    if st.button("➕ New Conversation", use_container_width=True):
        create_new_chat()
        st.rerun()
    
    st.markdown("### History")
    
    # Load and display conversations
    chats = db.get_all_conversations()
    for chat in chats:
        label = chat['title']
        if st.session_state.current_chat_id == chat['id']:
            label = f"🟢 {label}"
            
        if st.button(label, key=f"chat_{chat['id']}", use_container_width=True):
            switch_chat(chat['id'])
            st.rerun()

    if st.session_state.current_chat_id:
        st.divider()
        if st.button("🗑️ Delete Current Chat", type="primary"):
            delete_current_chat()

# --- Main Chat Area ---

# 1. Ensure a chat is selected
if st.session_state.current_chat_id is None:
    if not chats:
        create_new_chat()
        st.rerun()
    else:
        st.title("Recursive Language Model")
        st.info("Please select a conversation from the sidebar or start a new one.")
        st.stop()

# 2. Load Messages from DB
messages = db.load_messages(st.session_state.current_chat_id)

# 3. Update RLM Context & Render Stats
if st.session_state.rlm_instance:
    st.session_state.rlm_instance.context = messages
    # Render stats now that we have the messages
    render_sidebar_stats(messages)

# 4. Display Chat History
st.header(f"{next((c['title'] for c in chats if c['id'] == st.session_state.current_chat_id), 'New Chat')}")

for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 5. Input & Execution
if prompt := st.chat_input("Type your message..."):
    if not st.session_state.rlm_instance:
        st.error("Please enter an API Key in the sidebar.")
        st.stop()

    # A. Display User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # B. Save User Message
    db.save_message(st.session_state.current_chat_id, "user", prompt, tokens=len(prompt)//4)
    
    # Update sidebar immediately with new user message
    messages.append({"role": "user", "content": prompt, "tokens": len(prompt)//4})
    render_sidebar_stats(messages)

    # C. Generate Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        status_container = st.status("RLM is thinking recursively...", expanded=True)
        final_text = ""

        try:
            # Sync context before run
            st.session_state.rlm_instance.context = db.load_messages(st.session_state.current_chat_id)
            
            for step in st.session_state.rlm_instance.run(prompt):
                step_type = step.get("type")
                content = step.get("content")

                if step_type == "thought":
                    status_container.markdown("### 🧠 Thinking")
                    status_container.markdown(content)
                    status_container.markdown("---")
                elif step_type == "observation":
                    status_container.markdown("### 🛠️ Code Output")
                    status_container.code(content)
                    status_container.markdown("---")
                elif step_type == "final":
                    final_text = content
                    status_container.update(label="Response Generated", state="complete", expanded=False)
                elif step_type == "error":
                    status_container.error(f"Error: {content}")
                    status_container.update(state="error")

            # D. Save & Refresh
            if final_text:
                st.markdown(final_text)
                db.save_message(st.session_state.current_chat_id, "assistant", final_text, tokens=len(final_text)//4)
                
                # Update sidebar final time
                messages = db.load_messages(st.session_state.current_chat_id)
                render_sidebar_stats(messages)
                
                time.sleep(0.1) 
                st.rerun()

        except Exception as e:
            st.error(f"Runtime Exception: {e}")