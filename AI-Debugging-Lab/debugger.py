"""
debugger.py
-----------
The "brain" of the app. Each function:
  1. retrieves relevant context with RAG,
  2. fills in a prompt from prompts.py,
  3. asks Ollama,
  4. returns (answer_text, warning_or_None).

Nothing here ever executes the user's code - it is only sent to the LLM as text.
"""
import ollama_client
import prompts
import rag_pipeline
from ollama_client import OllamaError

TOP_K = 4  # how many knowledge chunks to retrieve


def _get_context(query, language, embedding_model):
    """Try RAG. If it fails, continue without context but report a warning."""
    try:
        results = rag_pipeline.retrieve(query, language, embedding_model, top_k=TOP_K)
        return rag_pipeline.format_context(results), None
    except (rag_pipeline.RAGError, OllamaError) as exc:
        return ("No reference material available.",
                f"Knowledge-base lookup failed, so the answer uses the model's general knowledge only. ({exc})")


def _ask(model, user_prompt):
    return ollama_client.chat(model, prompts.SYSTEM_PROMPT, user_prompt)


def analyze_code(code, language, model, embedding_model):
    # Use the start of the code as the search query (keeps embedding input small).
    context, warning = _get_context(f"{language} {code[:1000]}", language, embedding_model)
    prompt = prompts.DEBUG_PROMPT.format(language=language, context=context, code=code)
    return _ask(model, prompt), warning


def explain_error(error_text, language, model, embedding_model):
    context, warning = _get_context(f"{language} {error_text}", language, embedding_model)
    prompt = prompts.EXPLAIN_ERROR_PROMPT.format(language=language, context=context, error=error_text)
    return _ask(model, prompt), warning


def generate_test_cases(code, language, model, embedding_model):
    context, warning = _get_context(f"{language} test cases edge cases {code[:800]}", language, embedding_model)
    prompt = prompts.TEST_CASE_PROMPT.format(language=language, context=context, code=code)
    return _ask(model, prompt), warning


def generate_viva_question(code, language, model, previous_questions):
    previous = "\n".join(f"- {q}" for q in previous_questions) or "None yet."
    prompt = prompts.VIVA_QUESTION_PROMPT.format(language=language, code=code, previous=previous)
    return _ask(model, prompt), None


def evaluate_viva_answer(code, question, answer, language, model, embedding_model):
    context, warning = _get_context(f"{language} {question}", language, embedding_model)
    prompt = prompts.VIVA_EVALUATE_PROMPT.format(
        language=language, code=code, question=question, answer=answer, context=context)
    return _ask(model, prompt), warning
