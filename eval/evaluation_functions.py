"""Helpers for the DeepEval run: a judge-model wrapper and retrieval-rank metrics."""

import json
import re

import numpy as np
from deepeval.models import DeepEvalBaseLLM
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel


class LangChainJudge(DeepEvalBaseLLM):
    """Lets DeepEval use any LangChain chat model (here Gemini) as the LLM judge."""

    def __init__(self, llm, name: str = "langchain-judge"):
        self.model = llm
        self.name = name

    def load_model(self):
        return self.model

    def get_model_name(self) -> str:
        return self.name

    def generate(self, prompt: str, schema: BaseModel | None = None):
        if schema is None:
            return (self.model | StrOutputParser()).invoke(prompt)
        try:
            return self.model.with_structured_output(schema).invoke(prompt)
        except Exception:
            return self._parse_json(self.generate(prompt), schema)

    async def a_generate(self, prompt: str, schema: BaseModel | None = None):
        if schema is None:
            return await (self.model | StrOutputParser()).ainvoke(prompt)
        try:
            return await self.model.with_structured_output(schema).ainvoke(prompt)
        except Exception:
            return self._parse_json(await self.a_generate(prompt), schema)

    @staticmethod
    def _parse_json(text: str, schema: BaseModel):
        """Fallback: pull the first JSON object out of a plain-text reply."""
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return schema(**json.loads(match.group(0)))


def hit_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    """1.0 if any of the top-k retrieved chunks comes from a gold call, else 0.0."""
    return float(any(i in gold_ids for i in retrieved_ids[:k]))


def reciprocal_rank(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    """1 / rank of the first retrieved chunk from a gold call (0.0 if none)."""
    for rank, i in enumerate(retrieved_ids, start=1):
        if i in gold_ids:
            return 1.0 / rank
    return 0.0


def answer_similarity_metric(answer: str, golden_answer: str, embedder) -> float:
    """Cosine similarity between the embeddings of the answer and the golden answer.

    `embedder` is any LangChain `Embeddings` object, e.g. `rag.store.embeddings`.
    """
    va, vb = (np.asarray(v) for v in embedder.embed_documents([answer, golden_answer]))
    similarity =  float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))
    return similarity



