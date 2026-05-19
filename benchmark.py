import json
import os
import glob
from datetime import datetime

import openai

BENCHMARK_DIR = "./benchmark"

OPENAI_API_KEY = "sk-b93f045d7963409ba614d956d02a9269"
OPENAI_BASE_URL = "https://api.deepseek.com"
SCORING_MODEL = "deepseek-v4-flash"

S1 = """类别一： S1 预训练记忆干扰。
1. 答案中的数字、期限、专有名词与知识库原文完全一致。
2. 答案不包含知识库之外的事实性信息。
3. 答案提供可验证的证据引用。
"""
S2 = """
类别二： S2 无中生有。
1. 当知识库中不存在相关依据时，模型能够明确拒答。
2. 答案中不包含任何猜测、臆断或常识性补充。
3. 不得以建议性表述替代拒答结论。
"""
S3 = """
类别三： S3 忠实度错误。
1. 答案完整保留原文中的限制条件、适用前提和例外情形。
2. 答案不得改变原文的约束强度和语义方向，例如不得将“可以”改写为“必须”，或将“不得”改写为“原则上不建议”。
"""
S4 = """
类别四： S4 多跳推理错误。
1. 答案所依据的证据链完整，无关键证据缺失。
2. 答案能够清晰呈现跨条款、跨条件之间的推理桥接关系。
3. 不得仅凭局部证据直接给出最终结论。
"""

SCORING_CRITERIA = {
    "1": S1,
    "2": S2,
    "3": S3,
    "4": S4
}

class Benchmark:
    def __init__(self, answer_func, result_dir="./result"):
        self.get_answer = answer_func
        self.result_dir = result_dir
        self.data = self._load_data()
        self.client = openai.OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL
        )

    def _load_data(self):
        json_files = glob.glob(os.path.join(BENCHMARK_DIR, "*.json"))
        if not json_files:
            raise FileNotFoundError(f"在 {BENCHMARK_DIR}/ 目录下未找到 JSON 文件")
        target_file = json_files[0]
        print(f"加载评测数据: {target_file}")
        with open(target_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def run(self):
        print("\n" + "=" * 60)
        print("Benchmark 评测开始")
        print("=" * 60)

        results = self._qa_phase()

        print("\n" + "=" * 60)
        print("评分环节 (1=正确, 0=错误)")
        print("=" * 60)
        self._scoring_phase(results)

        print("\n" + "=" * 60)
        print("评测结果")
        print("=" * 60)

        total, correct, acc, per_label = self._compute_stats(results)
        print(f"总题数 {total} | 正确 {correct} | 准确率 {acc:.2%}")
        for label in SCORING_CRITERIA:
            s = per_label[label]
            print(f"  类别{label}: {s['total']}题 | 正确{s['correct']} | 准确率{s['accuracy']:.2%}")

        self._save_results(results)

        output = {
            "accuracy": acc,
            "correct": correct,
            "total": total,
            "results": results
        }
        for label in SCORING_CRITERIA:
            output[f"accuracy_{label}"] = per_label[label]["accuracy"]
            output[f"correct_{label}"]  = per_label[label]["correct"]
            output[f"total_{label}"]    = per_label[label]["total"]
        return output

    def _qa_phase(self):
        results = []
        total = len(self.data)

        for i, item in enumerate(self.data):
            label = item["label"]
            question = item["question"]
            ref = item["reference_answer"]

            print(f"\n[{i+1}/{total}] 类别 {label}: {question}")
            print(f"正确答案: {ref}")

            llm_answer, rag_context = self.get_answer(question)
            llm_answer = llm_answer.strip()
            print(f"LLM回答: {llm_answer}")

            results.append({
                "id": item["id"],
                "label": label,
                "question": question,
                "reference_answer": ref,
                "llm_answer": llm_answer,
                "rag_context": rag_context,
                "user_score": None
            })

        return results

    def _scoring_phase(self, results):
        print("AI 评分中...")
        total = len(results)
        for i, r in enumerate(results):
            print(f"\n[{i+1}/{total}] 类别:{r['label']} | ID:{r['id']}")
            score = self._ai_score(
                r["question"], r["reference_answer"],
                r["llm_answer"], r["rag_context"], r["label"]
            )
            r["user_score"] = score
            print(f"问题: {r['question']}")
            print(f"正确答案: {r['reference_answer']}")
            print(f"LLM回答: {r['llm_answer']}")
            print(f"AI评分: {'正确' if score == 1 else '错误'}")

    def _ai_score(self, question, reference_answer, llm_answer, context, label):
        criteria = SCORING_CRITERIA.get(label, "")
        prompt = f"""请根据以下评分标准判断LLM回答是否正确。

评分标准：
{criteria}

RAG检索到的上下文：
{context}

问题：{question}
正确答案：{reference_answer}
LLM回答：{llm_answer}

注意：请确保你的回答必须只有一个数字：1表示正确，0表示错误："""

        response = self.client.chat.completions.create(
            model=SCORING_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            extra_body={"thinking": {"type": "enabled"}}
        )
        print(f"response:{response}")
        result = response.choices[0].message.content.strip()
        print(f"评分AI回答:{result}")
        return 1 if "1" in result else 0

    def _compute_stats(self, results):
        total = len(results)
        correct = sum(1 for r in results if r["user_score"] == 1)
        acc = correct / total if total > 0 else 0

        per_label = {}
        for label in SCORING_CRITERIA:
            items = [r for r in results if r["label"] == label]
            t = len(items)
            c = sum(1 for r in items if r["user_score"] == 1)
            a = c / t if t > 0 else 0
            per_label[label] = {"total": t, "correct": c, "accuracy": a}

        return total, correct, acc, per_label

    def _save_results(self, results):
        os.makedirs(self.result_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.result_dir, f"benchmark_{timestamp}.json")

        total, correct, acc, per_label = self._compute_stats(results)

        output = {
            "accuracy": acc,
            "correct": correct,
            "total": total,
        }
        for label in SCORING_CRITERIA:
            output[f"accuracy_{label}"] = per_label[label]["accuracy"]
            output[f"correct_{label}"]  = per_label[label]["correct"]
            output[f"total_{label}"]    = per_label[label]["total"]
        output["results"] = results

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"\n结果已保存到 {filepath}")