# src/generator.py
import os
import subprocess
from typing import List

try:
    import openai
except Exception:
    openai = None

from transformers import pipeline

class Generator:
    def __init__(self, mode="openai", openai_api_key=None, hf_model="google/flan-t5-small", ollama_model="mistral:7b"):
        """
        Generator for RAG-based chatbot.
        mode: "openai", "hf", or "ollama"
        """
        self.mode = mode
        self.ollama_model = ollama_model

        if mode == "openai":
            if openai is None:
                raise RuntimeError("openai package required for mode=openai")
            if openai_api_key is None:
                openai_api_key = os.getenv("OPENAI_API_KEY")
            openai.api_key = openai_api_key

        elif mode == "hf":
            self.generator = pipeline(
                "text2text-generation",
                model=hf_model,
                device=0 if self._cuda_available() else -1
            )

        elif mode == "ollama":
            if not self._ollama_installed():
                raise RuntimeError("Ollama not installed or not in PATH. Install from https://ollama.ai")
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def _cuda_available(self):
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def _ollama_installed(self):
        try:
            subprocess.run(["ollama", "--version"], capture_output=True)
            return True
        except FileNotFoundError:
            return False

    def _generate_ollama(self, prompt: str) -> str:
        """
        Run Ollama model locally via subprocess and return full text output.
        """
        try:
            result = subprocess.run(
                ["ollama", "run", self.ollama_model, "--prompt", prompt],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            return f"Error running Ollama: {e.stderr}"

    def generate(self, user_query: str, contexts: List[str], max_tokens=256):
        """
        Generate answer using selected backend.
        contexts: list of retrieved chunks; will be concatenated into the prompt
        """
        ctx_text = "\n\n---\n\n".join(contexts)
        prompt = (
            f"You are a medical assistant. Use the context provided to answer the question in a detailed, patient-friendly way."
            f"If the context does not contain relevant information, reply: 'I am not sure, please consult a doctor.'"
            f"Include full sentences and explanations, avoid one-word answers.\n\n"
            f"Context:\n{ctx_text}\n\nQuestion: {user_query}\n\nAnswer:"
        )
        print(prompt)
        if self.mode == "openai":
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0
            )
            return resp["choices"][0]["message"]["content"].strip()


        elif self.mode == "hf":
            out = self.generator(
                prompt,
                max_new_tokens=max_tokens,
                min_length=150,  # <- force longer answers
                do_sample=True,  # <- allow richer outputs
                temperature=0.7,
                top_p=0.9
            )[0]["generated_text"]
            return out.strip()

        elif self.mode == "ollama":
            return self._generate_ollama(prompt)

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
