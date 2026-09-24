"""
groq_generator.py
-------------------
Generates a natural-language answer from retrieved context chunks
using Groq's hosted LLM API (Llama models, fast inference, no local
model download or GPU required — the model runs on Groq's servers).

Requires a free API key from https://console.groq.com/keys

Common Groq model names (check console.groq.com/docs/models for the
current list):
    llama-3.3-70b-versatile   - strong general-purpose model
    llama-3.1-8b-instant      - fastest, lighter weight
    gemma2-9b-it              - Google's Gemma, lightweight
"""

from typing import List, Dict
from groq import Groq


class GroqGenerator:
    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-oss-120b",
        temperature: float = 0.2,
    ):
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to a .env file or "
                "export it as an environment variable."
            )
        self.client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def _build_messages(self, question: str, matches: List[Dict]) -> List[Dict]:
        context_blocks = []
        for i, match in enumerate(matches, start=1):
            context_blocks.append(
                f"[Source {i}: {match['source']}, page {match['page']}]\n{match['text']}"
            )
        context = "\n\n".join(context_blocks)

        system_prompt = (
            "You are a helpful assistant answering questions using only the "
            "provided context. If the answer is not contained in the context, "
            "say \"I don't have enough information to answer that.\" "
            "Cite sources using their [Source N] label when relevant."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def generate(self, question: str, matches: List[Dict]) -> str:
        """
        Args:
            question: the user's natural language question
            matches: retrieved chunks from PineconeStore.query()
                     [{"text", "source", "page", "score"}, ...]

        Returns:
            Generated answer string.
        """
        if not matches:
            return "I don't have enough information to answer that."

        messages = self._build_messages(question, matches)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()

    def generate_stream(self, question: str, matches: List[Dict]):
        """
        Streaming version - yields answer tokens as they're generated.
        Useful for a UI with a live-typing effect.
        """
        if not matches:
            yield "I don't have enough information to answer that."
            return

        messages = self._build_messages(question, matches)

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=True,
        )
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield token
