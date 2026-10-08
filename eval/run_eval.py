"""Run the RAG pipeline over the gold dataset and score it with DeepEval.

Usage (from the project root, with the app stopped so Qdrant isn't locked):
    python -m eval.run_eval
"""

import json
import os
from pathlib import Path
from statistics import mean

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from deepeval.dataset import Golden
from deepeval.metrics import ContextualRecallMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

import src.config as config
from eval.evaluation_functions import LangChainJudge, hit_at_k, reciprocal_rank
from src.llm_client import get_llm
from src.rag import RAG

EVAL_DIR = Path("eval")
GOLD_PATH = EVAL_DIR / "gold_dataset.json"
RESULTS_PATH = EVAL_DIR / "results.json"
K_VALUES = range(1, config.RERANK_TOP_N + 1)


def run_llm_metric(metric, test_case) -> dict:
    """Score one test case; a judge/API failure is recorded instead of aborting the run."""
    try:
        metric.measure(test_case)
        return {"score": metric.score, "reason": metric.reason}
    except Exception as exc:
        return {"score": None, "reason": f"ERROR: {exc}"}


def main() -> None:
    # Built directly (not via EvaluationDataset loaders, whose signature differs across deepeval versions).
    goldens = [Golden(**row) for row in json.loads(GOLD_PATH.read_text(encoding="utf-8"))]

    judge = LangChainJudge(get_llm(), name=config.LLM_MODEL)
    recall = ContextualRecallMetric(model=judge, async_mode=False, include_reason=True)
    faithfulness = FaithfulnessMetric(model=judge, async_mode=False, include_reason=True)

    rag = RAG()
    results = []
    try:
        for i, golden in enumerate(goldens, start=1):
            print(f"[{i}/{len(goldens)}] {golden.input}")
            out = rag.rag_chain(golden.input)
            docs = out["documents"]
            retrieved_ids = [d.metadata.get("file_name") for d in docs]
            gold_ids = golden.additional_metadata["reference_context_ids"]

            test_case = LLMTestCase(
                input=golden.input,
                actual_output=out["answer"],
                expected_output=golden.expected_output,
                retrieval_context=[d.page_content for d in docs],
            )

            results.append(
                {
                    "input": golden.input,
                    "gold_ids": gold_ids,
                    "retrieved_ids": retrieved_ids,
                    "answer": out["answer"],
                    "contextual_recall": run_llm_metric(recall, test_case),
                    "faithfulness": run_llm_metric(faithfulness, test_case),
                    "hit_at_k": {k: hit_at_k(retrieved_ids, gold_ids, k) for k in K_VALUES},
                    "reciprocal_rank": reciprocal_rank(retrieved_ids, gold_ids),
                }
            )
    finally:
        # Local Qdrant allows one client per folder; release it even if the run fails,
        # otherwise a re-run in the same notebook kernel hits the lock error.
        rag.store.client.close()

    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFull results (answers, judge reasons) saved to {RESULTS_PATH}")


# def fmt(value) -> str:
#     return "  n/a" if value is None else f"{value:5.2f}"


# def print_report(results: list[dict]) -> None:
#     k_cols = "".join(f"  hit@{k}" for k in K_VALUES)
#     print(f"\n{'#':>2}  recall  faith{k_cols}    RR  question")
#     for i, r in enumerate(results, start=1):
#         hits = "".join(f"  {r['hit_at_k'][k]:5.0f}" for k in K_VALUES)
#         print(
#             f"{i:>2}  {fmt(r['contextual_recall']['score'])}  {fmt(r['faithfulness']['score'])}"
#             f"{hits}  {r['reciprocal_rank']:4.2f}  {r['input'][:60]}"
#         )

#     def avg(scores):
#         scores = [s for s in scores if s is not None]
#         return mean(scores) if scores else None

#     print("\nSummary")
#     print(f"  Contextual recall : {fmt(avg(r['contextual_recall']['score'] for r in results))}")
#     print(f"  Faithfulness      : {fmt(avg(r['faithfulness']['score'] for r in results))}")
#     for k in K_VALUES:
#         print(f"  Hit@{k}            : {fmt(avg(r['hit_at_k'][k] for r in results))}")
#     print(f"  MRR               : {fmt(avg(r['reciprocal_rank'] for r in results))}")


if __name__ == "__main__":
    main()
