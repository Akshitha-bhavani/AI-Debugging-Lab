"""
app.py
------
Streamlit user interface for AI Debugging Lab.
Run with:  streamlit run app.py
"""
import streamlit as st

import debugger
import rag_pipeline
from ollama_client import ModelNotInstalled, OllamaError, OllamaNotRunning, list_models
from rag_pipeline import RAGError

st.set_page_config(page_title="AI Debugging Lab", page_icon="🐞", layout="wide")


# ---------- Helpers ----------
def show_error(exc):
    """Show a friendly message for each kind of failure."""
    if isinstance(exc, OllamaNotRunning):
        st.error("🔌 Ollama is not running. Start it in a terminal, then try again:")
        st.code("ollama serve", language="bash")
    elif isinstance(exc, ModelNotInstalled):
        st.error(f"📦 {exc} Install it with:")
        st.code(f"ollama pull {exc.model}", language="bash")
    else:
        st.error(f"⚠️ {exc}")


def run_task(message, func, *args):
    """Run a function with a spinner; show errors instead of crashing. Returns None on failure."""
    try:
        with st.spinner(message):
            return func(*args)
    except (OllamaError, RAGError) as exc:
        show_error(exc)
    except Exception as exc:  # last safety net so the app never shows a traceback
        st.error(f"Unexpected error: {exc}")
    return None


def show_result(result):
    """result is (text, warning) or None."""
    if result is None:
        return
    text, warning = result
    if warning:
        st.warning(warning)
    st.markdown(text)


# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")
    language = st.selectbox("Programming language", ["Python", "C", "Java"])
    llm_model = st.text_input("Ollama model", value="llama3.2:3b")
    embed_model = st.text_input("Embedding model", value="nomic-embed-text")

    # Ollama connection status
    try:
        installed = list_models()
        st.success("Ollama: running")
        st.caption("Installed: " + (", ".join(installed) if installed else "none"))
    except OllamaError:
        st.error("Ollama: not reachable")

    # RAG status
    st.subheader("📚 Knowledge Base")
    status = rag_pipeline.get_status()
    if status["built"]:
        st.success(f"Ready ({status['chunks']} chunks)")
        st.caption(f"Embedded with: {status['embedding_model']}")
    else:
        st.info("Not built yet. It is built automatically on first use.")

    if st.button("🔄 Rebuild Knowledge Base"):
        count = run_task("Embedding knowledge base...", rag_pipeline.build_index, embed_model)
        if count:
            st.success(f"Built {count} chunks.")
            st.rerun()

    st.subheader("ℹ️ About")
    st.caption("AI Debugging Lab uses a local LLM (Ollama) and RAG to explain and debug code. "
               "It never executes your code - it only reads it.")

# ---------- Header ----------
st.title("🐞 AI Debugging Lab")
st.subheader("AI-Powered Code Debugging and Learning Assistant")

tab_debug, tab_error, tab_tests, tab_viva = st.tabs(
    ["🐞 Debug Code", "⚠️ Explain Error", "🧪 Test Cases", "🎤 Viva Practice"])

# ---------- Tab 1: Debug Code ----------
with tab_debug:
    st.caption(f"Language: **{language}** (change it in the sidebar)")
    code = st.text_area("Paste your code", height=280, key="debug_code",
                        placeholder="Paste your code here...")
    if st.button("🔍 Analyze Code"):
        if not code.strip():
            st.warning("Please paste some code first.")
        else:
            show_result(run_task("Analyzing code...", debugger.analyze_code,
                                 code, language, llm_model, embed_model))

# ---------- Tab 2: Explain Error ----------
with tab_error:
    error_text = st.text_area("Paste an error message", height=120, key="error_text",
                              placeholder="IndexError: list index out of range")
    if st.button("💡 Explain Error"):
        if not error_text.strip():
            st.warning("Please paste an error message first.")
        else:
            show_result(run_task("Explaining error...", debugger.explain_error,
                                 error_text, language, llm_model, embed_model))

# ---------- Tab 3: Test Cases ----------
with tab_tests:
    test_code = st.text_area("Paste a function or program", height=250, key="test_code",
                             placeholder="def add(a, b):\n    return a + b")
    if st.button("🧪 Generate Test Cases"):
        if not test_code.strip():
            st.warning("Please paste some code first.")
        else:
            show_result(run_task("Generating test cases...", debugger.generate_test_cases,
                                 test_code, language, llm_model, embed_model))

# ---------- Tab 4: Viva Practice ----------
with tab_viva:
    # session_state keeps values between Streamlit reruns
    st.session_state.setdefault("viva_question", None)
    st.session_state.setdefault("viva_code", "")
    st.session_state.setdefault("viva_history", [])

    viva_code = st.text_area("Paste your code", height=220, key="viva_code_input",
                             placeholder="Paste the code you want to be quizzed on...")
    if st.button("🎤 Generate Question"):
        if not viva_code.strip():
            st.warning("Please paste some code first.")
        else:
            result = run_task("Thinking of a question...", debugger.generate_viva_question,
                              viva_code, language, llm_model, st.session_state.viva_history)
            if result:
                st.session_state.viva_question = result[0]
                st.session_state.viva_code = viva_code
                st.session_state.viva_history.append(result[0])
                st.session_state.pop("viva_answer", None)  # clear the old answer box

    if st.session_state.viva_question:
        st.info(f"**Question:** {st.session_state.viva_question}")
        answer = st.text_area("Your answer", height=150, key="viva_answer")
        if st.button("📊 Evaluate Answer"):
            if not answer.strip():
                st.warning("Please type your answer first.")
            else:
                show_result(run_task("Evaluating your answer...", debugger.evaluate_viva_answer,
                                     st.session_state.viva_code, st.session_state.viva_question,
                                     answer, language, llm_model, embed_model))
