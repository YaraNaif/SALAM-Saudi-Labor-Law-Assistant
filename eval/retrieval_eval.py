import json
from pathlib import Path
from typing import List, Dict
import math
import matplotlib.pyplot as plt

from app.retriever import Retriever
from app.config import CONFIG


def load_eval_queries(path: Path) -> List[Dict]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def reciprocal_rank(predicted_articles: List[str], gold_articles: List[str]) -> float:
    """
    Return 1/rank_of_first_gold if found, else 0.
    Ranks are 1-based.
    """
    gold_set = set(gold_articles)
    for idx, art_id in enumerate(predicted_articles, start=1):
        if art_id in gold_set:
            return 1.0 / idx
    return 0.0


def hit_at_k(predicted_articles: List[str], gold_articles: List[str], k: int) -> float:
    """
    Return 1.0 if any gold article appears in top-k predictions, else 0.0.
    """
    gold_set = set(gold_articles)
    for art_id in predicted_articles[:k]:
        if art_id in gold_set:
            return 1.0
    return 0.0


def ndcg_at_k(predicted_articles: List[str], gold_articles: List[str], k: int) -> float:
    """
    Treat each gold article as relevance=1, non-gold as 0.
    DCG = sum_{i=1..k} (rel_i / log2(i+1))
    IDCG = ideal DCG (all relevant at top)

    Since we assume binary relevance, IDCG is basically:
      if we have G golds:
          IDCG = sum_{i=1..min(G,k)} 1/log2(i+1)
    """
    gold_set = set(gold_articles)
    dcg = 0.0
    for i, art_id in enumerate(predicted_articles[:k], start=1):
        rel = 1.0 if art_id in gold_set else 0.0
        dcg += rel / math.log2(i + 1.0)

    # ideal DCG
    G = min(len(gold_set), k)
    idcg = 0.0
    for i in range(1, G + 1):
        idcg += 1.0 / math.log2(i + 1.0)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def evaluate_retrieval(eval_data: List[Dict]):
    retriever = Retriever()

    results = []
    all_hits_at_1 = []
    all_hits_at_3 = []
    all_hits_at_5 = []
    all_mrr = []
    all_ndcg5 = []
    first_correct_ranks = []  # for histogram/debug

    for ex in eval_data:
        question = ex["question"]
        gold = ex["gold_articles"]  # list[str], e.g. ["Article 75"]

        ctx_chunks = retriever.get_context(question)
        predicted_ids = [c.article_id for c in ctx_chunks]

        # metrics
        h1 = hit_at_k(predicted_ids, gold, 1)
        h3 = hit_at_k(predicted_ids, gold, 3)
        h5 = hit_at_k(predicted_ids, gold, 5)

        rr = reciprocal_rank(predicted_ids, gold)
        ndcg5 = ndcg_at_k(predicted_ids, gold, 5)

        all_hits_at_1.append(h1)
        all_hits_at_3.append(h3)
        all_hits_at_5.append(h5)
        all_mrr.append(rr)
        all_ndcg5.append(ndcg5)

        # first-correct rank for histogram
        rank_found = None
        for idx, art_id in enumerate(predicted_ids, start=1):
            if art_id in set(gold):
                rank_found = idx
                break
        first_correct_ranks.append(rank_found if rank_found is not None else 999)

        results.append({
            "question": question,
            "gold": gold,
            "predicted": predicted_ids,
            "hit@1": h1,
            "hit@3": h3,
            "hit@5": h5,
            "reciprocal_rank": rr,
            "ndcg@5": ndcg5,
            "first_correct_rank": rank_found,
        })

    # aggregate
    summary = {
        "hit@1": sum(all_hits_at_1) / len(all_hits_at_1),
        "hit@3": sum(all_hits_at_3) / len(all_hits_at_3),
        "hit@5": sum(all_hits_at_5) / len(all_hits_at_5),
        "mrr":   sum(all_mrr)       / len(all_mrr),
        "ndcg@5":sum(all_ndcg5)     / len(all_ndcg5),
        "num_examples": len(eval_data),
    }

    return results, summary, first_correct_ranks


def plot_retrieval_summary(summary: Dict):
    """
    Bar plot of Hit@1, Hit@3, Hit@5, MRR, NDCG@5
    """
    metrics = ["hit@1", "hit@3", "hit@5", "mrr", "ndcg@5"]
    values = [summary[m] for m in metrics]

    plt.figure()
    plt.bar(metrics, values)
    plt.title("Retrieval Quality Summary (higher is better)")
    plt.ylim(0, 1.05)
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    plt.ylabel("Score")
    plt.show()


def plot_rank_hist(first_correct_ranks: List[int]):
    """
    Histogram of the first correct article's rank.
    1 means perfect. 999 means we never found a correct article.
    """
    plt.figure()
    plt.hist(first_correct_ranks, bins=[1,2,3,4,5,6,10,999], align="left", rwidth=0.8)
    plt.title("Distribution of Rank of First Correct Article")
    plt.xlabel("Rank (999 = not found)")
    plt.ylabel("#queries")
    plt.show()


def main():
    eval_path = Path("eval/qa_eval.jsonl")
    eval_data = load_eval_queries(eval_path)

    results, summary, first_correct_ranks = evaluate_retrieval(eval_data)

    print("=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\n=== PER-QUESTION ===")
    for r in results:
        print(json.dumps(r, ensure_ascii=False, indent=2))

    plot_retrieval_summary(summary)
    plot_rank_hist(first_correct_ranks)


if __name__ == "__main__":
    main()