"""
ollama_client.py
----------------
All communication with the local Ollama server lives here.
We use plain `requests` calls to Ollama's REST API, so there is no extra SDK to learn.
Custom exceptions let the UI show friendly messages instead of crashing.
"""
import os
import requests

# Ollama listens on this address by default. Override with the OLLAMA_HOST env variable.
_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
BASE_URL = _host if _host.startswith("http") else f"http://{_host}"


# ---------- Custom exceptions (the UI catches these) ----------
class OllamaError(Exception):
    """Base class for all Ollama-related problems."""


class OllamaNotRunning(OllamaError):
    """Ollama server cannot be reached."""


class ModelNotInstalled(OllamaError):
    """The requested model has not been pulled."""

    def __init__(self, model):
        self.model = model
        super().__init__(f"Model '{model}' is not installed in Ollama.")


class OllamaTimeout(OllamaError):
    """The request took too long."""


class InvalidResponse(OllamaError):
    """Ollama replied, but the reply was empty or malformed."""


# ---------- Low-level helper ----------
def _post(path, payload, timeout):
    """Send a POST request and translate common failures into our exceptions."""
    try:
        response = requests.post(f"{BASE_URL}{path}", json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError:
        raise OllamaNotRunning("Cannot connect to Ollama. Start it with `ollama serve`.")
    except requests.exceptions.Timeout:
        raise OllamaTimeout(f"Ollama did not answer within {timeout} seconds. "
                            "Try a smaller model or shorter code.")
    return response


# ---------- Public functions ----------
def list_models():
    """Return the names of installed models. Raises OllamaNotRunning if offline."""
    try:
        response = requests.get(f"{BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]
    except requests.exceptions.ConnectionError:
        raise OllamaNotRunning("Cannot connect to Ollama. Start it with `ollama serve`.")
    except (requests.exceptions.RequestException, ValueError, KeyError) as exc:
        raise InvalidResponse(f"Could not read the model list: {exc}")


def chat(model, system_prompt, user_prompt, temperature=0.2, timeout=180):
    """Ask the LLM a question and return its answer as text."""
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        # Low temperature = more factual, less creative answers.
        "options": {"temperature": temperature, "num_ctx": 4096},
    }
    response = _post("/api/chat", payload, timeout)

    if response.status_code == 404:
        raise ModelNotInstalled(model)
    if response.status_code != 200:
        raise InvalidResponse(f"Ollama returned HTTP {response.status_code}: {response.text[:200]}")

    try:
        text = response.json()["message"]["content"].strip()
    except (ValueError, KeyError):
        raise InvalidResponse("Ollama sent a response in an unexpected format.")
    if not text:
        raise InvalidResponse("The model returned an empty answer. Please try again.")
    return text


def embed(model, texts, timeout=120):
    """Turn a list of texts into a list of embedding vectors."""
    response = _post("/api/embed", {"model": model, "input": texts}, timeout)

    if response.status_code == 404:
        if "model" in response.text.lower():
            raise ModelNotInstalled(model)
        return _embed_legacy(model, texts, timeout)  # very old Ollama versions
    if response.status_code != 200:
        raise InvalidResponse(f"Embedding request failed (HTTP {response.status_code}).")

    try:
        vectors = response.json()["embeddings"]
    except (ValueError, KeyError):
        raise InvalidResponse("Embedding response was in an unexpected format.")
    if len(vectors) != len(texts):
        raise InvalidResponse("Number of embeddings does not match number of texts.")
    return vectors


def _embed_legacy(model, texts, timeout):
    """Fallback for older Ollama versions that only have /api/embeddings."""
    vectors = []
    for text in texts:
        response = _post("/api/embeddings", {"model": model, "prompt": text}, timeout)
        if response.status_code == 404:
            raise ModelNotInstalled(model)
        try:
            vectors.append(response.json()["embedding"])
        except (ValueError, KeyError):
            raise InvalidResponse("Embedding response was in an unexpected format.")
    return vectors
