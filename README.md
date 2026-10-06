# 🐞 AI Debugging Lab
**AI-Powered Code Debugging and Learning Assistant**

A Streamlit web app that helps students understand and debug Python, C and Java code using a **local LLM (Ollama)** and a **RAG pipeline** over a small programming knowledge base. Everything runs on your own computer: no paid APIs, no API keys.

> ⚠️ **Safety:** The app **never executes** submitted code. It only sends the code as text to the LLM, which reads it and reasons about it. AI answers can be wrong, so always verify fixes by running the corrected code yourself in a safe environment.

## Features
- 🐞 **Debug Code** – identifies the issue type (syntax / runtime / logical), explains it, and gives corrected code
- ⚠️ **Explain Error** – explains any error message with an example, a fix and prevention tips
- 🧪 **Test Cases** – generates normal, boundary, edge and invalid cases in a table
- 🎤 **Viva Practice** – asks one question at a time, then scores your answer out of 10
- 📚 **RAG** – answers are grounded in `knowledge_base/` files
- 🛡️ Friendly handling of Ollama being offline, missing models, empty input, timeouts and vector-store problems

## Technology Stack
Python · Streamlit · Ollama (`llama3.2:3b`, `nomic-embed-text`) · FAISS (`faiss-cpu`) · NumPy · Requests

(LangChain is intentionally **not** used: the RAG pipeline is short enough to write directly, which makes it easier to understand and explain.)

## System Architecture
```
 Streamlit UI (app.py)
        │
 debugger.py ── builds prompts (prompts.py)
        │              │
        │        rag_pipeline.py ──► FAISS index (vector_store/)
        │              │                    ▲
        │              └── embeddings ──────┘ (Ollama: nomic-embed-text)
        ▼
 ollama_client.py ──► Ollama LLM (llama3.2:3b) ──► answer shown in UI
```

## How RAG Works Here
1. **Document files** – `knowledge_base/*.txt` hold short sections about errors and concepts.
2. **Text extraction** – each file is read as plain text.
3. **Chunking** – text is split on blank lines; small sections are merged to ~800 characters.
4. **Embeddings** – each chunk is turned into a vector by `nomic-embed-text`.
5. **Vector database** – vectors are stored in a FAISS index saved in `vector_store/`.
6. **Similarity search** – the user's code/error is embedded and compared to all chunks (cosine similarity).
7. **Relevant context** – the top 4 chunks (for the chosen language + general concepts) are selected.
8. **Ollama** – the chunks are placed in the prompt together with the user's input.
9. **Final answer** – the LLM replies, using the context where relevant.

Only the top chunks are sent to the model, never the whole knowledge base. If RAG fails, the app shows a warning and still answers from the model's general knowledge.

## Installation
**Prerequisites:** Python 3.9+, [Ollama](https://ollama.com/download) installed.

```bash
# 1. (Recommended) create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

# 2. Install Python packages
pip install -r requirements.txt

# 3. Download the models
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

## Running the Project
```bash
ollama serve            # skip if Ollama already runs in the background
streamlit run app.py
```
Open the URL Streamlit prints (usually http://localhost:8501). The knowledge base is built automatically the first time you analyze something; you can also click **Rebuild Knowledge Base** in the sidebar (do this after editing the knowledge files).

## Example Usage
Paste this Python code into **Debug Code**:
```python
numbers = [10, 20, 30]
for i in range(len(numbers) + 1):
    print(numbers[i])
```
Expected response (wording will vary): issue type **Runtime error**; problem: the loop reaches index 3 but the list only has indexes 0–2; cause: `range(len(numbers) + 1)` is one too large, so Python raises `IndexError: list index out of range`; corrected code uses `range(len(numbers))` or `for n in numbers:`; plus a beginner explanation and one tip (e.g. "prefer iterating directly over a list"). The response must not claim the code was run.

## Project Structure
```
AI-Debugging-Lab/
├── app.py                 # Streamlit UI (sidebar + 4 tabs)
├── debugger.py            # Feature logic: RAG lookup + prompt + Ollama call
├── ollama_client.py       # HTTP calls to Ollama, custom exceptions
├── rag_pipeline.py        # Chunking, embeddings, FAISS index, retrieval
├── prompts.py             # All prompt templates
├── requirements.txt
├── README.md
├── .gitignore
├── knowledge_base/        # python_errors, c_errors, java_errors, programming_concepts
└── vector_store/          # generated FAISS index (git-ignored)
```

## Common Errors and Solutions
| Problem | Solution |
|---|---|
| "Ollama is not running" | Run `ollama serve` in another terminal |
| "Model ... is not installed" | `ollama pull llama3.2:3b` / `ollama pull nomic-embed-text` |
| `faiss` import error | `pip install faiss-cpu` (use Python 3.9–3.12 if wheels are unavailable) |
| Request times out | Use a smaller model, paste less code, close other heavy apps |
| Warning: knowledge-base lookup failed | Check the embedding model is installed, then click Rebuild Knowledge Base |
| Answers are weak or wrong | Try a larger model (e.g. `llama3.1:8b`) and type its name in the sidebar |

## Limitations
- Code is **not executed**, so outputs and test-case expectations are the model's reasoning, not verified results
- Small local models can make mistakes or ignore the output format
- Only Python, C and Java; small knowledge base; no authentication or history storage
- Very long code may exceed the model's context window

## Future Enhancements
- Safe sandboxed execution (e.g. in Docker) to verify fixes and test cases
- Support for more languages (JavaScript, C++)
- Upload `.py/.c/.java` files and PDF/Markdown knowledge documents
- Syntax-highlighted editor and side-by-side diff of original vs corrected code
- Viva score history and progress chart
- Streaming responses and a chat-style follow-up mode

## Resume Description
**AI Debugging Lab – AI-Powered Code Debugging and Learning Assistant** (Python, Streamlit, Ollama, FAISS, RAG)
Built a fully local web application that helps students debug Python, C and Java code. Implemented a Retrieval-Augmented Generation pipeline (chunking, `nomic-embed-text` embeddings, FAISS similarity search) to ground a local `llama3.2:3b` model in a curated error knowledge base. Designed reusable prompt templates, modular architecture, robust error handling, and four tools: code debugging, error explanation, test-case generation and an interactive viva evaluator, with no paid APIs and no execution of untrusted code.
