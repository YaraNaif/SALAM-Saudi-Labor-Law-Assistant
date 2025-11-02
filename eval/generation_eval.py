import json
import re
from pathlib import Path
from typing import List, Dict

from app.retriever import Retriever
from app.llm_engine import LLMEngine


def load_eval_queries(path: Path) -> List[Dict]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def extract_section(answer: str, section_label: str) -> str:
    """
    Pull the body of a numbered section from the model answer.

    We split on lines that begin with "1." / "2." / "3." / "4.".
    Then we match the section number we care about (like "2.").
    """
    parts = re.split(r"(?m)^\s*(\d\.\s)", answer)
    # parts looks like: ["", "1. ", "body1...", "2. ", "body2...", ...]
    pairs = []
    for i in range(1, len(parts), 2):
        label = parts[i].strip()
        body = parts[i + 1] if (i + 1) < len(parts) else ""
        pairs.append((label, body.strip()))

    want_num = section_label.split(".", 1)[0].strip()
    for (lbl, body) in pairs:
        if lbl.startswith(want_num):
            return body
    return ""


def find_article_mentions(section_text: str) -> List[str]:
    """
    Return all article numbers explicitly mentioned.
    We normalize both Arabic ("المادة 75") and English ("Article 75") forms.
    Output is a list of the numbers as strings, e.g. ["75", "77"].
    """
    out = []
    # English style: Article 75
    for m in re.finditer(r"[Aa]rticle\s+(\d+)", section_text):
        out.append(m.group(1))
    # Arabic style: المادة 75
    for m in re.finditer(r"المادة\s+(\d+)", section_text):
        out.append(m.group(1))

    # Deduplicate
    return list(set(out))


def normalize_gold_articles(gold_list: List[str]) -> List[str]:
    """
    gold_list like ["Article 75"] -> ["75"]
    """
    nums = []
    for g in gold_list:
        m = re.search(r"(\d+)", g)
        if m:
            nums.append(m.group(1))
    return nums


def evaluate_generation(eval_data: List[Dict]):
    retriever = Retriever()
    llm = LLMEngine()

    rows = []
    valid_structure_count = 0
    correct_citation_count = 0

    # For classification metrics
    TP = 0
    FP = 0
    FN = 0
    TN = 0

    for ex in eval_data:
        q = ex["question"]
        gold_articles = ex["gold_articles"]  # ["Article 75", ...]
        gold_nums = normalize_gold_articles(gold_articles)

        # retrieve
        ctx_chunks = retriever.get_context(q)

        # generate
        ans = llm.generate_answer(q, ctx_chunks)

        # structural check
        has_sec1 = "1." in ans
        has_sec2 = "2." in ans
        has_sec3 = "3." in ans
        has_sec4 = "4." in ans
        full_structure_ok = has_sec1 and has_sec2 and has_sec3 and has_sec4
        if full_structure_ok:
            valid_structure_count += 1

        # citation extraction from section 2
        sec2 = extract_section(ans, "2.")
        mentioned_nums = find_article_mentions(sec2)

        # determine if model cited correctly
        # citation_ok = True if any mentioned_nums overlap with gold_nums
        overlap = any(num in gold_nums for num in mentioned_nums)
        if overlap:
            correct_citation_count += 1

        # classification accounting:
        #
        # We assume every eval item requires a correct legal article
        # because your dataset is all labor-law questions.
        #
        # So the "ground truth label" is always that there IS a correct
        # article to cite, and that article is gold_nums.
        #
        # - TP: model cited AND it matched gold
        # - FP: model cited BUT it did NOT match gold
        # - FN: model did NOT cite anything matching gold
        # - TN: (basically never happens in this dataset,
        #        but we'll include the logic anyway in case you later add
        #        questions where no article applies)
        #
        model_cited_anything = (len(mentioned_nums) > 0)
        gold_available = (len(gold_nums) > 0)

        if gold_available:
            if overlap:
                TP += 1
            else:
                # model tried but was wrong -> FP
                # OR model said nothing -> FN
                if model_cited_anything:
                    FP += 1
                else:
                    FN += 1
        else:
            # gold_available == False (not in our current dataset, but safe code)
            if model_cited_anything:
                FP += 1
            else:
                TN += 1

        rows.append({
            "question": q,
            "gold_articles": gold_articles,
            "generated_answer": ans,
            "structure_ok": full_structure_ok,
            "gold_nums": gold_nums,
            "mentioned_nums": mentioned_nums,
            "citation_ok": overlap,
        })

    n = len(eval_data)
    structure_rate = valid_structure_count / n if n else 0.0
    citation_rate = correct_citation_count / n if n else 0.0

    # compute precision, recall, f1, accuracy safely
    tp_fp = TP + FP
    tp_fn = TP + FN
    denom_acc = TP + TN + FP + FN

    precision = (TP / tp_fp) if tp_fp > 0 else 0.0
    recall    = (TP / tp_fn) if tp_fn > 0 else 0.0
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    accuracy = ((TP + TN) / denom_acc) if denom_acc > 0 else 0.0

    summary = {
        "answers_with_full_structure_%": structure_rate,
        "answers_with_correct_citation_%": citation_rate,
        "num_examples": n,

        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,

        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }

    return rows, summary


def main():
    eval_path = Path("eval/qa_eval.jsonl")
    eval_data = load_eval_queries(eval_path)

    rows, summary = evaluate_generation(eval_data)

    print("=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\n=== PER-QUESTION ===")
    for r in rows:
        print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()