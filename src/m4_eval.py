from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                            context_precision, context_recall], progress_bar=False)
        try:
            df = result.to_pandas()
        except Exception:
            df = result
        per_question = []
        for row in df:
            per_question.append(EvalResult(
                question=row.get("question", ""),
                answer=row.get("answer", ""),
                contexts=row.get("contexts", []),
                ground_truth=row.get("ground_truth", ""),
                faithfulness=float(row.get("faithfulness", 0.0)),
                answer_relevancy=float(row.get("answer_relevancy", 0.0)),
                context_precision=float(row.get("context_precision", 0.0)),
                context_recall=float(row.get("context_recall", 0.0)),
            ))
        return {
            "faithfulness": float(result.aggregate.get("faithfulness", 0.0)) if hasattr(result, "aggregate") else float(df["faithfulness"].mean() if "faithfulness" in df.column_names else 0.0),
            "answer_relevancy": float(result.aggregate.get("answer_relevancy", 0.0)) if hasattr(result, "aggregate") else float(df["answer_relevancy"].mean() if "answer_relevancy" in df.column_names else 0.0),
            "context_precision": float(result.aggregate.get("context_precision", 0.0)) if hasattr(result, "aggregate") else float(df["context_precision"].mean() if "context_precision" in df.column_names else 0.0),
            "context_recall": float(result.aggregate.get("context_recall", 0.0)) if hasattr(result, "aggregate") else float(df["context_recall"].mean() if "context_recall" in df.column_names else 0.0),
            "per_question": per_question,
        }
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return {"faithfulness": 0.0, "answer_relevancy": 0.0,
                "context_precision": 0.0, "context_recall": 0.0, "per_question": []}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating or unsupported facts", "Tighten prompt, add grounding and citations"),
        "answer_relevancy": ("Answer doesn't match the question", "Improve prompt template or question understanding"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filtering"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or retrieval coverage"),
    }

    scored = []
    for item in eval_results:
        metrics = {
            "faithfulness": item.faithfulness,
            "answer_relevancy": item.answer_relevancy,
            "context_precision": item.context_precision,
            "context_recall": item.context_recall,
        }
        avg = sum(metrics.values()) / len(metrics)
        worst_metric = min(metrics, key=metrics.get)
        scored.append({
            "question": item.question,
            "answer": item.answer,
            "ground_truth": item.ground_truth,
            "worst_metric": worst_metric,
            "score": float(metrics[worst_metric]),
            "diagnosis": diagnostic_tree[worst_metric][0],
            "suggested_fix": diagnostic_tree[worst_metric][1],
            "metrics": metrics,
            "avg_score": avg,
        })

    scored.sort(key=lambda x: x["avg_score"])
    return scored[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
