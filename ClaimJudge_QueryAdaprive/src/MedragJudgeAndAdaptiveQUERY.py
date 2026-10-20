
"""
imedrag_claim_judge.py
=======================

یک پیاده‌سازی کاملاً مستقل و جدا از i-MedRAG (نسخه‌ی follow-up/iterative موجود در
medrag.py) که یک لایه‌ی "Judge" (داور) بر پایه‌ی ادعا (claim) اضافه می‌کند.

ایده:
    1) برای هر sub-query که در طول چرخه‌ی iterative تولید می‌شود، پاسخ خام از
       متد اصلی و بدون تغییر MedRAG.medrag_answer() گرفته می‌شود (فراخوانی
       می‌شود، نه بازنویسی).
    2) پاسخ خام به مجموعه‌ای از ادعاهای اتمی (atomic claims) شکسته می‌شود.
    3) هر ادعا به‌صورت مجزا در برابر اسناد بازیابی‌شده (retrieved documents)
       توسط یک judge LLM بررسی می‌شود که آیا آن ادعا از اسناد پشتیبانی
       می‌شود (entailed) یا نه.
    4) فقط ادعاهایی که judge آن‌ها را «درست/پشتیبانی‌شده» تشخیص می‌دهد وارد
       History (متغیر context در حلقه‌ی iterative) می‌شوند؛ ادعاهای رد شده
       دور ریخته می‌شوند.


"""

import os
import re
import json
import sys


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.append(_THIS_DIR)

from liquid import Template  

from medrag import MedRAG


jmr_claim_extraction_system = (
    "You are a meticulous fact-decomposition assistant. Your only task is to "
    "break a given answer into a list of short, atomic, independently-checkable "
    "factual claims. Do not add information that is not present in the answer. "
    "Do not omit important claims. Each claim must be a single, self-contained "
    "statement (no pronouns without a clear referent). Respond with JSON only."
)

jmr_claim_extraction_prompt = Template(
    """Here is an answer produced by an assistant:

{{answer}}

Decompose this answer into a list of atomic factual claims. Each claim should
be short (one sentence), self-contained, and checkable on its own against a
set of reference documents.

Respond ONLY with JSON in this exact format, and nothing else:
{"claims": ["claim 1", "claim 2", "..."]}"""
)

jmr_claim_verification_system = (
    "You are a fact-checking judge for a medical question-answering system. "
    "Your job is to catch claims that are FALSE according to the reference "
    "documents -- NOT to demand that every claim be repeated word-for-word "
    "from the source text. Most claims that are true but phrased differently, "
    "more specific, more general, or synthesized from multiple documents "
    "should NOT be rejected.\n\n"
    "Use exactly one of these three verdicts:\n"
    "- 'contradicted': the documents state something that directly conflicts "
    "with this claim, OR the claim misattributes/fabricates what a specific "
    "document says.\n"
    "- 'supported': the documents state this, clearly imply it, or it is a "
    "reasonable and consistent synthesis/paraphrase of what they say.\n"
    "- 'unverifiable': the documents simply do not address this topic at all "
    "(neither confirming nor denying it). This is NOT an error and is not "
    "the same as false -- plenty of true, reasonable clinical statements are "
    "outside the scope of a given document excerpt.\n\n"
    "Only 'contradicted' claims will be discarded downstream, so reserve it "
    "for claims that are actually wrong, not merely unproven. When genuinely "
    "unsure between 'supported' and 'unverifiable', prefer 'unverifiable' -- "
    "never default to 'contradicted' out of caution. Respond with JSON only."
)

jmr_claim_verification_prompt = Template(
    """Reference documents:
{{documents}}

Claim to verify:
{{claim}}

Does the claim directly conflict with the documents above ('contradicted'), 
is it stated or reasonably implied by them ('supported'), or do the documents 
simply not address it either way ('unverifiable')?

Respond ONLY with JSON in this exact format, and nothing else:
{"verdict": "supported" or "contradicted" or "unverifiable", "reason": "brief justification"}"""
)


jmr_plan_prompt = (
    "You are planning the retrieval budget for a medical question. Decide how many "
    "search queries to send to a medical textbook retriever in this round.\n\n"
    "{context_block}{question_prompt}\n\n"
    "Rules:\n"
    "- One query = one distinct piece of knowledge that is still missing (for example a "
    "disease definition, a mechanism, a drug indication, a diagnostic criterion).\n"
    "- Do NOT plan one query per answer option. Most questions can be solved with 1 or 2 "
    "queries about the key concept.\n"
    "- If one query would retrieve two facts together, count them as ONE.\n"
    "- Simple question or one missing concept: 1. Two separate missing facts: 2. Several "
    "unrelated facts or multi-step reasoning: 3 or more.\n"
    "{zero_rule}"
    "- Never exceed {max_q}.\n\n"
    "Output JSON only, nothing else:\n"
    '{{"n_queries": <integer from {min_q} to {max_q}>, "missing": "<one short sentence: '
    'what is still unknown>"}}'
)

jmr_adaptive_queries_suffix = (
    "\n\nIMPORTANT: Ask EXACTLY {k} {noun} in the ## Queries section, no more and no fewer. "
    "Each query must target a different missing piece of knowledge{missing_part}. "
    "Do NOT ask one query per answer option. Keep exactly the same output format as "
    "requested above."
)

_JMR_META_CLAIM_PATTERNS = [
    re.compile(r"^documents?\s*\[\d+(\]|,|\s|and|\bexists?\b)", re.IGNORECASE),
    re.compile(r"^document\s*\[\d+\]\s+(exists|is\s+titled)", re.IGNORECASE),
    re.compile(r"^documents?\s*\[[\d,\s]+\]\s+(mention|discuss|is|are)\b", re.IGNORECASE),
]


def _jmr_is_meta_claim(claim_text):
    """True اگر ادعا صرفاً درباره‌ی وجود/عنوان یک Document باشد، نه محتوای بالینی."""
    stripped = claim_text.strip()
    return any(p.match(stripped) for p in _JMR_META_CLAIM_PATTERNS)


class ClaimJudgedIMedRAG(MedRAG):

    # ------------------------------------------------------------------
    # گام ۱: شکستن یک پاسخ به مجموعه‌ای از ادعاهای اتمی
    # ------------------------------------------------------------------
    def jmr_extract_claims(self, answer_text, **kwargs):
        """پاسخ خام را به لیستی از ادعاهای اتمی تجزیه می‌کند."""
        prompt = jmr_claim_extraction_prompt.render(answer=answer_text)
        messages = [
            {"role": "system", "content": jmr_claim_extraction_system},
            {"role": "user", "content": prompt},
        ]
        raw_output = self.generate(messages, **kwargs)  

        claims = self._jmr_parse_claims_json(raw_output)
        if claims is None:
            claims = [
                s.strip() for s in re.split(r"(?<=[.!?])\s+", answer_text)
                if s.strip()
            ]
        return claims

    @staticmethod
    def _jmr_parse_claims_json(raw_output):
        try:
            match = re.search(r"\{.*\}", raw_output, re.DOTALL)
            payload = match.group(0) if match else raw_output
            data = json.loads(payload)
            claims = data.get("claims", [])
            claims = [c.strip() for c in claims if isinstance(c, str) and c.strip()]
            return claims
        except Exception:
            return None

    # ------------------------------------------------------------------
    # گام ۲: بررسی یک ادعا در برابر اسناد بازیابی‌شده
    # ------------------------------------------------------------------
    def jmr_verify_claim(self, claim_text, documents_text, **kwargs):
        """
        برمی‌گرداند: (verdict: "supported"|"contradicted"|"unverifiable", reason: str)

        نکته‌ی مهم: فقط "contradicted" یعنی ادعا غلط است. "unverifiable" به
        معنای غلط بودن نیست -- یعنی سند درباره‌اش چیزی نگفته. هر دوی
        "supported" و "unverifiable" باید در History نگه داشته شوند؛ فقط
        "contradicted" حذف می‌شود.
        """
        prompt = jmr_claim_verification_prompt.render(
            documents=documents_text, claim=claim_text
        )
        messages = [
            {"role": "system", "content": jmr_claim_verification_system},
            {"role": "user", "content": prompt},
        ]
        raw_output = self.generate(messages, **kwargs)

        verdict, reason = self._jmr_parse_verdict_json(raw_output)
        if verdict not in ("supported", "contradicted", "unverifiable"):
            verdict = "unverifiable"
        return verdict, reason

    @staticmethod
    def _jmr_parse_verdict_json(raw_output):
        try:
            match = re.search(r"\{.*\}", raw_output, re.DOTALL)
            payload = match.group(0) if match else raw_output
            data = json.loads(payload)
            verdict = str(data.get("verdict", "")).strip().lower()
            reason = str(data.get("reason", "")).strip()
            return verdict, reason
        except Exception:
            lowered = raw_output.lower()
            if "contradicted" in lowered or "contradict" in lowered:
                return "contradicted", raw_output
            if "supported" in lowered:
                return "supported", raw_output
            return "unverifiable", raw_output

    # ------------------------------------------------------------------
    # گام ۳: اجرای یک sub-query کامل (retrieve + answer + decompose + judge)
    # ------------------------------------------------------------------
    def jmr_answer_subquery_with_judge(self, subquery_text, k=10, rrf_k=100, **kwargs):
        """
        یک sub-query را از طریق medrag_answer اصلی (بدون هیچ تغییری) پاسخ
        می‌دهد، سپس پاسخ را به ادعا می‌شکند و هر ادعا را در برابر همان اسناد
        بازیابی‌شده برای این sub-query داوری می‌کند.

        خروجی:
            verified_answer_text (str): فقط ادعاهای تاییدشده، به‌هم چسبانده
            claim_audit (list[dict]): ریز جزئیات هر ادعا برای شفافیت/دیباگ
            retrieved_snippets, scores: خروجی خام retrieval برای این sub-query
        """
        raw_answer, retrieved_snippets, scores = self.medrag_answer(
            subquery_text, k=k, rrf_k=rrf_k, **kwargs
        )

        if isinstance(raw_answer, list):
            raw_answer = raw_answer[0] if len(raw_answer) > 0 else ""
        documents_text = "\n".join(
            "Document [{:d}] (Title: {:s}) {:s}".format(
                idx, snip.get("title", ""), snip.get("content", "")
            )
            for idx, snip in enumerate(retrieved_snippets)
        )

        claims = self.jmr_extract_claims(raw_answer, **kwargs)
        content_claims = [c for c in claims if not _jmr_is_meta_claim(c)]

        claim_audit = []
        contradicted_claims = []
        kept_claims = []
        seen_normalized = set()  
        for claim in content_claims:
            verdict, reason = self.jmr_verify_claim(claim, documents_text, **kwargs)
            claim_audit.append(
                {"claim": claim, "verdict": verdict, "reason": reason}
            )
            if verdict == "contradicted":
                contradicted_claims.append(claim)
                continue
            # supported یا unverifiable -> نگه داشته می‌شود (فقط غلط حذف می‌شود)
            norm = re.sub(r"\W+", " ", claim.lower()).strip()
            if norm in seen_normalized:
                continue
            seen_normalized.add(norm)
            kept_claims.append(claim)

        if not contradicted_claims:
            verified_answer_text = raw_answer.strip()
        elif kept_claims:
            verified_answer_text = " ".join(kept_claims)
        else:
            verified_answer_text = ""

        return verified_answer_text, claim_audit, retrieved_snippets, scores

    # ------------------------------------------------------------------
    # planner: مدل تعداد query لازم برای این دور را انتخاب می‌کند
    # ------------------------------------------------------------------
    def jmr_plan_num_queries(
        self, question_prompt, context, max_queries, first_round, **kwargs
    ):
        min_q = 1 if first_round else 0
        context_block = (
            f"Information gathered so far:\n{context}\n\n" if context else ""
        )
        zero_rule = (
            "- You need at least 1 query in this round.\n"
            if first_round
            else "- If the information gathered so far is already enough to answer with "
            "confidence, output 0.\n"
        )
        prompt = jmr_plan_prompt.format(
            context_block=context_block,
            question_prompt=question_prompt,
            zero_rule=zero_rule,
            min_q=min_q,
            max_q=max_queries,
        )

        k = min(2, max_queries)  
        missing = ""
        try:
            raw = self.generate([{"role": "user", "content": prompt}], **kwargs)
            m = re.search(r"[\"']?n_queries[\"']?\s*:\s*[\"']?(\d+)", raw)
            k = int(m.group(1))
            mm = re.search(
                r"[\"']?missing[\"']?\s*:\s*[\"'](.*?)[\"']\s*[},]", raw, re.DOTALL
            )
            missing = mm.group(1).strip() if mm else ""
        except Exception as e:
            print(f"[Planner] could not parse plan, using default {k}: {e}")

        k = max(min_q, min(k, max_queries))
        return k, missing

    def jmr_build_follow_up_ask(self, k, missing="", adaptive=True):
        """
        متن اصلی follow_up_ask را (بدون تغییر) با k می‌سازد. اگر adaptive=True
        باشد، پسوندی اضافه می‌شود که «دقیقاً k query» و «نه یکی برای هر گزینه» را
        می‌خواهد و آنچه هنوز نامعلوم است را یادآوری می‌کند.
        """
        base = self.templates["follow_up_ask"].format(k)
        if adaptive:
            noun = "query" if k == 1 else "queries"
            missing_part = f" (still missing: {missing})" if missing else ""
            base += jmr_adaptive_queries_suffix.format(
                k=k, noun=noun, missing_part=missing_part
            )
        return base

    def judged_i_medrag_answer(
        self,
        question,
        options=None,
        k=10,
        rrf_k=100,
        save_path=None,
        n_rounds=4,
        n_queries=3,
        qa_cache_path=None,
        adaptive_queries=True,
        **kwargs
    ):

        if options is not None:
            options_text = "\n".join(
                key + ". " + options[key] for key in sorted(options.keys())
            )
        else:
            options_text = ""
        question_prompt = f"Here is the question:\n{question}\n\n{options_text}"

        context = ""
        jmr_audit_log = [] 
        if save_path:

            open(save_path + ".verified.txt", "w").close()
        qa_cache = []
        if qa_cache_path is not None and os.path.exists(qa_cache_path):
            qa_cache = eval(open(qa_cache_path, "r").read())[:n_rounds]
            if len(qa_cache) > 0:
                context = qa_cache[-1]
            n_rounds = n_rounds - len(qa_cache)
        last_context = None

        fixed_ask_text = self.jmr_build_follow_up_ask(n_queries, adaptive=False)
        plan_k = n_queries
        self.jmr_queries_per_round = []
        self.jmr_plan_log = []
        jmr_asked_norm = set()  

        max_iterations = n_rounds + 3
        saved_messages = [
            {"role": "system", "content": self.templates["i_medrag_system"]}
        ]

        for i in range(max_iterations):
            force_answer = False
            if adaptive_queries and i < n_rounds:
                plan_k, plan_missing = self.jmr_plan_num_queries(
                    question_prompt,
                    context,
                    n_queries,
                    first_round=(context == ""),
                    **kwargs,
                )
                self.jmr_plan_log.append(
                    {"round": i + 1, "n_queries": plan_k, "missing": plan_missing}
                )
                print(f"[Round {i + 1}] planner chose {plan_k} quer(ies): {plan_missing}")
                if plan_k == 0:
                    force_answer = True
                    self.jmr_queries_per_round.append(0)
                    n_rounds = i
                    ask_text = ""
                else:
                    ask_text = self.jmr_build_follow_up_ask(
                        plan_k, plan_missing, adaptive=True
                    )
            else:
                ask_text = fixed_ask_text

            if i < n_rounds:
                if context == "":
                    messages = [
                        {"role": "system", "content": self.templates["i_medrag_system"]},
                        {
                            "role": "user",
                            "content": f"{question_prompt}\n\n{ask_text}",
                        },
                    ]
                else:
                    messages = [
                        {"role": "system", "content": self.templates["i_medrag_system"]},
                        {
                            "role": "user",
                            "content": f"{context}\n\n{question_prompt}\n\n{ask_text}",
                        },
                    ]
            elif force_answer or context != last_context:
                messages = [
                    {"role": "system", "content": self.templates["i_medrag_system"]},
                    {
                        "role": "user",
                        "content": f"{context}\n\n{question_prompt}\n\n{self.templates['follow_up_answer']}",
                    },
                ]
            elif len(messages) == 1:
                messages = [
                    {"role": "system", "content": self.templates["i_medrag_system"]},
                    {
                        "role": "user",
                        "content": f"{context}\n\n{question_prompt}\n\n{self.templates['follow_up_answer']}",
                    },
                ]

            saved_messages.append(messages[-1])
            if save_path:
                with open(save_path, "w") as f:
                    json.dump(
                        [p if type(p) == dict else p.model_dump() for p in saved_messages],
                        f,
                        indent=4,
                    )

            last_context = context
            last_content = self.generate(messages, **kwargs)
            response_message = {"role": "assistant", "content": last_content}
            saved_messages.append(response_message)
            if save_path:
                with open(save_path, "w") as f:
                    json.dump(
                        [p if type(p) == dict else p.model_dump() for p in saved_messages],
                        f,
                        indent=4,
                    )

            is_answer_round = i >= n_rounds and (
                "## Answer" in last_content or "answer is" in last_content.lower()
            )

            if is_answer_round:
                messages.append(response_message)
                messages.append(
                    {
                        "role": "user",
                        "content": "Output the answer in JSON: {'answer': your_answer (A/B/C/D)}"
                        if options
                        else "Output the answer in JSON: {'answer': your_answer}",
                    }
                )
                saved_messages.append(messages[-1])
                answer_content = self.generate(messages, **kwargs)
                answer_message = {"role": "assistant", "content": answer_content}
                messages.append(answer_message)
                saved_messages.append(messages[-1])
                if save_path:
                    with open(save_path, "w", encoding="utf-8") as f:
                        json.dump(
                            [p if type(p) == dict else p.model_dump() for p in saved_messages],
                            f,
                            indent=4,
                            ensure_ascii=False,
                        )
                    with open(save_path + ".claims.json", "w", encoding="utf-8") as f:
                        json.dump(jmr_audit_log, f, indent=4, ensure_ascii=False)

                return messages[-1]["content"], messages, jmr_audit_log

            elif "## Queries" in last_content:
                messages = messages[:-1]
                if last_content.split("## Queries")[-1].strip() == "":
                    print("Empty queries. Continue with next iteration.")
                    continue
                try:
                    action_str = self.generate(
                        [
                            {
                                "role": "user",
                                "content": (
                                    f"Parse the following passage and extract the queries as a list: {last_content}.\n\n"
                                    "Present the queries as they are. DO NOT merge or break down queries. "
                                    'Output the list of queries in JSON format: {"output": ["query 1", ..., "query N"]}'
                                ),
                            }
                        ],
                        **kwargs,
                    )
                    action_match = re.search(r"output\":\s*(\[.*\])", action_str, re.DOTALL)
                    action_str = action_match.group(1)
                    action_list = [
                        re.sub(r"^\d+\.\s*", "", s.strip()) for s in eval(action_str)
                    ]
                except Exception as e:
                    print("Error parsing action list. Continue with next iteration.")
                    error_class = e.__class__.__name__
                    error = f"{error_class}: {str(e)}"
                    print(error)
                    if save_path:
                        with open(save_path + ".error", "a") as f:
                            f.write(f"{error}\n")
                    continue

                action_list = [q for q in action_list if q.strip() != ""]
                if adaptive_queries:
                    unique_queries = []
                    for q in action_list:
                        key = re.sub(r"\W+", " ", q.lower()).strip()
                        if key and key not in jmr_asked_norm:
                            jmr_asked_norm.add(key)
                            unique_queries.append(q)
                    action_list = unique_queries[:plan_k]
                self.jmr_queries_per_round.append(len(action_list))
                print(
                    f"[Round {i + 1}] queries used: {len(action_list)} "
                    f"(planned {plan_k}, max {n_queries})"
                )

                for subquery in action_list:
                    if subquery.strip() == "":
                        continue
                    try:
                        (
                            verified_answer_text,
                            claim_audit,
                            _retrieved_snippets,
                            _scores,
                        ) = self.jmr_answer_subquery_with_judge(
                            subquery, k=k, rrf_k=rrf_k, **kwargs
                        )
                        jmr_audit_log.append(
                            {"round": i + 1, "query": subquery, "claims": claim_audit}
                        )
                        if save_path:
                            with open(save_path + ".claims.json", "w", encoding="utf-8") as f:
                                json.dump(jmr_audit_log, f, indent=4, ensure_ascii=False)
                            ok = [
                                c["claim"] for c in claim_audit
                                if c.get("verdict") != "contradicted"
                            ]
                            dropped = [
                                c["claim"] for c in claim_audit
                                if c.get("verdict") == "contradicted"
                            ]
                            with open(save_path + ".verified.txt", "a", encoding="utf-8") as f:
                                f.write(f"[Round {i + 1}] Query: {subquery}\n")
                                if ok:
                                    for claim in ok:
                                        f.write(f"  - {claim}\n")
                                else:
                                    f.write("  (no claim verified)\n")
                                if dropped:
                                    f.write("  [removed as contradicted]\n")
                                    for claim in dropped:
                                        f.write(f"    x {claim}\n")
                                f.write("\n")
                        if verified_answer_text.strip() == "":
                            continue
                        context += f"\n\nQuery: {subquery}\nAnswer: {verified_answer_text}"
                        context = context.strip()
                    except Exception as e:
                        error_class = e.__class__.__name__
                        error = f"{error_class}: {str(e)}"
                        print(error)
                        if save_path:
                            with open(save_path + ".error", "a") as f:
                                f.write(f"{error}\n")

                qa_cache.append(context)
                if qa_cache_path:
                    with open(qa_cache_path, "w") as f:
                        json.dump(qa_cache, f, indent=4)
            else:
                messages.append(response_message)
                print("No queries or answer. Continue with next iteration.")
                continue

        return messages[-1]["content"], messages, jmr_audit_log