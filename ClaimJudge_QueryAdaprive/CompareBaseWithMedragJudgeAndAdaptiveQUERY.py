
"""
run_adaptive_vs_imedrag.py

مقایسه‌ی i-MedRAG اصلی (medrag.py، baseline) با نسخه‌ی Claim-Judged
adaptive (claimverfierv2judge.py) روی سوالات test.json.


خروجی:
claim_comparison_adaptive/
  question_0001_xxxxxxxx/
    baseline/  messages.json, qa_cache.json, final_answer.txt
    claim/     messages.json, qa_cache.json, final_answer.txt, claims.json,
               all_claims.json, verified_claims.json, verified_claims.txt,
               queries.json
    comparison.json
  dataset_comparison.json, kept_claims_dataset.json, kept_claims_dataset.txt


"""

import os
import json
import re
import hashlib
from pathlib import Path
import sys

sys.path.append("src")

from MedragJudgeAndAdaptiveQUERY import ClaimJudgedIMedRAG
from medrag import MedRAG


# ============================================================
# تنظیمات
# ============================================================

TEST_FILE = "test.json"
OUTPUT_DIR = "claim_comparison_query"

# LLM_NAME = "ArvanCloud/GPT-4o-mini"
LLM_NAME = "Chabokan/meta-llama/llama-3.1-8b-instruct"
RAG = True
FOLLOW_UP = True

RETRIEVER_NAME = "MedCPT"
CORPUS_NAME = "Textbooks"
DB_DIR = "./corpus"

K = 10
RRF_K = 100
N_ROUNDS = 2

# سقف تعداد query در هر دور برای نسخه‌ی adaptive (مدل ۱ تا این عدد انتخاب می‌کند)
N_QUERIES_MAX = 4
# تعداد query که baseline در هر دور می‌خواهد (ثابت)
BASELINE_N_QUERIES = N_QUERIES_MAX

# True: verified_claims.* فقط claimهای "supported" (برای بازبینی انسانی)
# False: "unverifiable" هم می‌آید (هرچه در History نگه داشته شده)
SAVE_ONLY_SUPPORTED = True


# ============================================================
# ابزارهای فایل
# ============================================================

def load_questions(path):
    """هم JSONL و هم JSON array را می‌خواند."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("test.json باید list از سوالات باشد.")
        return data

    questions = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            questions.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"خط {line_no} در {path} JSON معتبر نیست: {e}")
    return questions


def safe_question_id(index, question):
    digest = hashlib.md5(question.encode("utf-8")).hexdigest()[:8]
    return f"question_{index:04d}_{digest}"


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def save_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text if text else "")


# ============================================================
# استخراج پاسخ مدل
# ============================================================

def extract_answer_from_json(text, options=None):
    """
    Extract answer choice (A-D) from model output.
    Supports {"answer": "D"}, {'answer': 'D'}, answer: D, The answer is D,
    and full option text.
    """
    if not text:
        return None

    text = text.strip()

    # 1. Standard JSON
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "answer" in data:
            answer = str(data["answer"]).strip()
            if re.fullmatch(r"[A-Da-d]", answer):
                return answer.upper()
            if options:
                for key, value in options.items():
                    if answer.lower() == str(value).strip().lower():
                        return key.upper()
    except Exception:
        pass

    # 2. Python-dict style: {'answer': 'D'}
    match = re.search(
        r"""['"]answer['"]\s*:\s*['"]([A-Da-d])['"]""", text, re.IGNORECASE
    )
    if match:
        return match.group(1).upper()

    # 3. answer: D
    match = re.search(r"\banswer\s*:\s*([A-Da-d])\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 4. The answer is D
    match = re.search(
        r"\b(?:the\s+)?(?:correct\s+)?answer\s+is\s*:?\s*([A-Da-d])\b",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()

    # 5. Full option text
    if options:
        text_lower = text.lower()
        for key, value in options.items():
            option_text = str(value).strip().lower()
            if option_text and option_text in text_lower:
                return key.upper()

    # 6. Last fallback: standalone A/B/C/D
    matches = re.findall(r"\b([A-Da-d])\b", text)
    if len(matches) == 1:
        return matches[0].upper()

    return None


def is_correct_prediction(prediction, gold):
    if prediction is None or gold is None:
        return False
    return str(prediction).strip().upper() == str(gold).strip().upper()


# ============================================================
# شمارش query ها
# ============================================================

def count_baseline_queries(cache_path):
    """
    baseline (medrag.py) تعداد query را جدا برنمی‌گرداند، ولی qa_cache.json
    آخرین context را نگه می‌دارد که هر query در آن به شکل
    'Query: ...' در ابتدای یک خط آمده است.
    """
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data:
            return 0
        return len(re.findall(r"(?m)^Query: ", data[-1]))
    except Exception:
        return 0


def list_claim_queries(audit_log):
    """query های واقعاً اجراشده‌ی نسخه‌ی Claim (هر entry در audit_log یک query است)."""
    return [
        item.get("query", "")
        for item in (audit_log or [])
        if item.get("query")
    ]


# ============================================================
# ذخیره Claimها (verdict سه‌حالته)
# ============================================================

def save_verified_claims(question_dir, audit_log):
    """
    هر entry از audit_log: {"round": r, "query": "...", "claims": [...]}
    هر claim: {"claim": "...", "verdict": "supported"|"contradicted"|"unverifiable", "reason": "..."}

    برمی‌گرداند: (all_claims, kept_claims, contradicted_claims)
      kept_claims         = supported + unverifiable (آنچه در History می‌ماند)
      contradicted_claims = آنچه به‌عنوان غلط حذف شده
    """
    question_dir = Path(question_dir)

    all_claims = []
    kept_claims = []
    contradicted_claims = []
    supported_only = []

    for round_item in audit_log or []:
        query = round_item.get("query", "")
        round_no = round_item.get("round")

        for item in round_item.get("claims", []):
            verdict = str(item.get("verdict", "")).strip().lower()

            if verdict not in ("supported", "contradicted", "unverifiable"):
                if "supported" in item:
                    verdict = "supported" if item.get("supported") else "contradicted"
                else:
                    verdict = "unverifiable"

            record = {
                "round": round_no,
                "query": query,
                "claim": item.get("claim", ""),
                "verdict": verdict,
                "reason": item.get("reason", ""),
            }

            all_claims.append(record)

            if verdict == "contradicted":
                contradicted_claims.append(record)
            else:
                kept_claims.append(record)
                if verdict == "supported":
                    supported_only.append(record)

    save_json(question_dir / "all_claims.json", all_claims)

    to_write = supported_only if SAVE_ONLY_SUPPORTED else kept_claims
    save_json(question_dir / "verified_claims.json", to_write)

    with (question_dir / "verified_claims.txt").open("w", encoding="utf-8") as f:
        if not to_write:
            f.write("(no verified claims)\n")
        else:
            for i, item in enumerate(to_write, start=1):
                f.write(f"{i}. [{item['verdict']}] {item['claim']}\n")
                f.write(f"   Query: {item['query']}\n")
                f.write(f"   Reason: {item['reason']}\n\n")

    return all_claims, kept_claims, contradicted_claims


# ============================================================
# اجرای یک سوال
# ============================================================

def run_one_question(index, item, baseline, claim_system, output_root):

    question = item["question"]
    options = item.get("options")

    qid = safe_question_id(index, question)
    qdir = Path(output_root) / qid
    baseline_dir = qdir / "baseline"
    claim_dir = qdir / "claim"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    claim_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 90)
    print(f"QUESTION {index}")
    print("=" * 90)
    print(question)

    # --------------------------------------------------------
    # 1. i-MedRAG اصلی (baseline)
    # --------------------------------------------------------
    print("\n[1/2] Running original i-MedRAG ...")

    baseline_save_path = baseline_dir / "messages.json"
    baseline_cache_path = baseline_dir / "qa_cache.json"

    try:
        baseline_answer, baseline_messages = baseline.i_medrag_answer(
            question=question,
            options=options,
            k=K,
            rrf_k=RRF_K,
            save_path=str(baseline_save_path),
            n_rounds=N_ROUNDS,
            n_queries=BASELINE_N_QUERIES,
            qa_cache_path=str(baseline_cache_path),
        )
        save_text(baseline_dir / "final_answer.txt", baseline_answer)
        baseline_error = None

    except Exception as e:
        baseline_answer = ""
        baseline_error = f"{type(e).__name__}: {e}"
        save_text(baseline_dir / "error.txt", baseline_error)
        print("[Baseline ERROR]", baseline_error)

    baseline_queries = count_baseline_queries(baseline_cache_path)

    # --------------------------------------------------------
    # 2. Claim-Judged adaptive
    # --------------------------------------------------------
    print("\n[2/2] Running Claim-Judged adaptive i-MedRAG ...")

    claim_save_path = claim_dir / "messages.json"
    claim_cache_path = claim_dir / "qa_cache.json"

    # اگر از اجرای قبلی cache مانده باشد، judged_i_medrag_answer از همان ادامه
    # می‌دهد و شمارش query غلط می‌شود؛ پس قبل از اجرا پاکش می‌کنیم.
    if claim_cache_path.exists():
        claim_cache_path.unlink()

    claim_system.jmr_queries_per_round = []
    claim_system.jmr_plan_log = []

    try:
        claim_answer, claim_messages, audit_log = claim_system.judged_i_medrag_answer(
            question=question,
            options=options,
            k=K,
            rrf_k=RRF_K,
            save_path=str(claim_save_path),
            n_rounds=N_ROUNDS,
            n_queries=N_QUERIES_MAX,
            qa_cache_path=str(claim_cache_path),
            adaptive_queries=True,
        )
        save_text(claim_dir / "final_answer.txt", claim_answer)
        save_json(claim_dir / "claims.json", audit_log)
        claim_error = None

    except Exception as e:
        claim_answer = ""
        audit_log = []
        claim_error = f"{type(e).__name__}: {e}"
        save_text(claim_dir / "error.txt", claim_error)
        print("[Claim ERROR]", claim_error)

    claim_query_list = list_claim_queries(audit_log)
    claim_queries = len(claim_query_list)
    queries_per_round = list(getattr(claim_system, "jmr_queries_per_round", []))
    save_json(
        claim_dir / "queries.json",
        {
            "queries_executed": claim_queries,
            "queries_per_round": queries_per_round,
            "planner_decisions": list(getattr(claim_system, "jmr_plan_log", [])),
            "queries": claim_query_list,
        },
    )

    # --------------------------------------------------------
    # 3. Claimها
    # --------------------------------------------------------
    all_claims, kept_claims, contradicted_claims = save_verified_claims(
        claim_dir, audit_log
    )

    supported_count = sum(1 for c in all_claims if c["verdict"] == "supported")
    unverifiable_count = sum(1 for c in all_claims if c["verdict"] == "unverifiable")
    contradicted_count = len(contradicted_claims)
    kept = len(kept_claims)

    # --------------------------------------------------------
    # 4. Prediction و درست/غلط
    # --------------------------------------------------------
    expected_answer = item.get("answer")
    expected_idx = item.get("answer_idx")

    baseline_pred = extract_answer_from_json(baseline_answer, options)
    claim_pred = extract_answer_from_json(claim_answer, options)

    base_ok = is_correct_prediction(baseline_pred, expected_idx)
    gate_ok = is_correct_prediction(claim_pred, expected_idx)

    # --------------------------------------------------------
    # 5. Comparison
    # --------------------------------------------------------
    comparison = {
        "question_id": qid,
        "question": question,
        "expected_answer": expected_answer,
        "expected_answer_idx": expected_idx,
        "baseline": {
            "prediction": baseline_pred,
            "raw_final_answer": baseline_answer,
            "error": baseline_error,
            "queries_used": baseline_queries,
        },
        "claim_judged": {
            "prediction": claim_pred,
            "raw_final_answer": claim_answer,
            "error": claim_error,
            "queries_used": claim_queries,
            "queries_per_round": queries_per_round,
            "total_claims": len(all_claims),
            "kept_claims": kept,
            "contradicted_claims": contradicted_count,
            "supported_claims": supported_count,
            "unverifiable_claims": unverifiable_count,
        },
        "prediction_changed": baseline_pred != claim_pred,
        "kept_claims_list": [x["claim"] for x in kept_claims],
        "contradicted_claims_list": [x["claim"] for x in contradicted_claims],
        "summary": {
            "gold": expected_idx,
            "base": baseline_pred,
            "gate": claim_pred,
            "base_ok": base_ok,
            "gate_ok": gate_ok,
            "base_queries": baseline_queries,
            "gate_queries": claim_queries,
            "queries_per_round": queries_per_round,
            "kept": kept,
            "supported": supported_count,
            "unverifiable": unverifiable_count,
            "contradicted": contradicted_count,
        },
    }

    save_json(qdir / "comparison.json", comparison)

    print("\n--- RESULT ---")
    print("Expected:", expected_idx, expected_answer)
    print("Baseline prediction:", baseline_pred, f"| queries: {baseline_queries}")
    print(
        "Claim-Judged prediction:",
        claim_pred,
        f"| queries: {claim_queries} (per round: {queries_per_round})",
    )
    print("Total claims:", len(all_claims))
    print(
        f"Kept (supported+unverifiable): {kept}  |  "
        f"Contradicted (removed): {contradicted_count}  |  "
        f"Supported: {supported_count}  |  Unverifiable: {unverifiable_count}"
    )

    return comparison


# ============================================================
# جدول نهایی
# ============================================================

def print_summary_table(results):
    """
    #   Gold  Base  Gate  BaseOK  GateOK  BaseQ  GateQ  PerRound  Kept  Contr  Unver
    """
    width = 96

    print("\n")
    print("=" * width)
    print("FINAL DATASET SUMMARY")
    print("=" * width)

    print(
        f"{'#':<4}"
        f"{'Gold':<6}"
        f"{'Base':<6}"
        f"{'Gate':<6}"
        f"{'BaseOK':<8}"
        f"{'GateOK':<8}"
        f"{'BaseQ':<7}"
        f"{'GateQ':<7}"
        f"{'PerRound':<12}"
        f"{'Kept':<6}"
        f"{'Contr':<7}"
        f"{'Unver':<6}"
    )
    print("-" * width)

    base_correct = 0
    gate_correct = 0
    base_q_total = 0
    gate_q_total = 0

    for i, result in enumerate(results, start=1):
        s = result.get("summary", {})

        base_ok = s.get("base_ok", False)
        gate_ok = s.get("gate_ok", False)
        base_q = s.get("base_queries", 0)
        gate_q = s.get("gate_queries", 0)
        per_round = ",".join(str(x) for x in s.get("queries_per_round", [])) or "-"

        base_correct += int(base_ok)
        gate_correct += int(gate_ok)
        base_q_total += base_q
        gate_q_total += gate_q

        print(
            f"{i:<4}"
            f"{str(s.get('gold')):<6}"
            f"{str(s.get('base')):<6}"
            f"{str(s.get('gate')):<6}"
            f"{str(base_ok):<8}"
            f"{str(gate_ok):<8}"
            f"{base_q:<7}"
            f"{gate_q:<7}"
            f"{per_round:<12}"
            f"{s.get('kept', 0):<6}"
            f"{s.get('contradicted', 0):<7}"
            f"{s.get('unverifiable', 0):<6}"
        )

    print("-" * width)

    total = len(results)
    if total > 0:
        print(
            f"Baseline accuracy:     {base_correct}/{total} "
            f"({100.0 * base_correct / total:.1f}%)"
        )
        print(
            f"Claim-Judged accuracy: {gate_correct}/{total} "
            f"({100.0 * gate_correct / total:.1f}%)"
        )
        print(
            f"\nQueries (total):   baseline = {base_q_total}   "
            f"adaptive = {gate_q_total}"
        )
        print(
            f"Queries (average): baseline = {base_q_total / total:.2f}   "
            f"adaptive = {gate_q_total / total:.2f}   (max per round = {N_QUERIES_MAX})"
        )
        if base_q_total > 0:
            saving = 100.0 * (base_q_total - gate_q_total) / base_q_total
            print(f"Adaptive uses {saving:.1f}% fewer queries than baseline")


# ============================================================
# main
# ============================================================

def main():

    questions = load_questions(TEST_FILE)
    if not questions:
        print(f"No questions found in {TEST_FILE}")
        return

    output_root = Path(OUTPUT_DIR)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(questions)} questions from {TEST_FILE}")

    print("\nInitializing original i-MedRAG ...")
    baseline = MedRAG(
        llm_name=LLM_NAME,
        rag=RAG,
        follow_up=FOLLOW_UP,
        retriever_name=RETRIEVER_NAME,
        corpus_name=CORPUS_NAME,
        db_dir=DB_DIR,
    )

    print("\nInitializing Claim-Judged adaptive i-MedRAG ...")
    claim_system = ClaimJudgedIMedRAG(
        llm_name=LLM_NAME,
        rag=RAG,
        follow_up=FOLLOW_UP,
        retriever_name=RETRIEVER_NAME,
        corpus_name=CORPUS_NAME,
        db_dir=DB_DIR,
    )

    results = []

    for index, item in enumerate(questions, start=1):
        try:
            results.append(
                run_one_question(
                    index=index,
                    item=item,
                    baseline=baseline,
                    claim_system=claim_system,
                    output_root=output_root,
                )
            )
        except Exception as e:
            print(
                f"\n[FATAL QUESTION ERROR] question={index}: "
                f"{type(e).__name__}: {e}"
            )

    save_json(output_root / "dataset_comparison.json", results)

    print_summary_table(results)

    # --------------------------------------------------------
    # Claimهای نگه‌داشته‌شده‌ی همه‌ی سوالات
    # --------------------------------------------------------
    kept_dataset = [
        {
            "question_id": r["question_id"],
            "question": r["question"],
            "expected_answer": r["expected_answer"],
            "expected_answer_idx": r["expected_answer_idx"],
            "queries_used": r["claim_judged"]["queries_used"],
            "kept_claims": r["kept_claims_list"],
            "contradicted_claims": r["contradicted_claims_list"],
        }
        for r in results
    ]
    save_json(output_root / "kept_claims_dataset.json", kept_dataset)

    with (output_root / "kept_claims_dataset.txt").open("w", encoding="utf-8") as f:
        for r in results:
            f.write("=" * 90 + "\n")
            f.write(f"QUESTION: {r['question_id']}\n")
            f.write(r["question"] + "\n")
            f.write(
                f"Expected answer: {r['expected_answer_idx']} - {r['expected_answer']}\n"
            )
            f.write(
                f"Queries used: baseline={r['baseline']['queries_used']}  "
                f"adaptive={r['claim_judged']['queries_used']} "
                f"(per round: {r['claim_judged']['queries_per_round']})\n\n"
            )

            if r["kept_claims_list"]:
                f.write("KEPT CLAIMS (supported + unverifiable):\n")
                for claim in r["kept_claims_list"]:
                    f.write(f"- {claim}\n")
            else:
                f.write("KEPT CLAIMS: none\n")

            if r["contradicted_claims_list"]:
                f.write("\nREMOVED AS CONTRADICTED:\n")
                for claim in r["contradicted_claims_list"]:
                    f.write(f"x {claim}\n")

            f.write("\n")

    print("\n" + "=" * 90)
    print("FINISHED")
    print("=" * 90)
    print(f"Output directory: {output_root.resolve()}")
    print(f"Questions processed: {len(results)}")
    print(f"Dataset comparison: {output_root / 'dataset_comparison.json'}")
    print(f"Kept claims dataset: {output_root / 'kept_claims_dataset.json'}")


if __name__ == "__main__":
    main()