import json
import os
import glob
from datetime import datetime

BENCHMARK_DIR = "./benchmark"
RESULT_DIR = "./result"

class Benchmark:
    def __init__(self, answer_func):
        self.get_answer = answer_func
        self.data = self._load_data()

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

        total = len(results)
        correct = sum(1 for r in results if r["user_score"] == 1)
        acc = correct / total if total > 0 else 0
        print(f"总题数 {total} | 正确 {correct} | 准确率 {acc:.2%}")

        self._save_results(results)

        return {
            "accuracy": acc,
            "correct": correct,
            "total": total,
            "results": results
        }

    def _qa_phase(self):
        results = []
        total = len(self.data)

        for i, item in enumerate(self.data):
            label = item["label"]
            question = item["question"]
            ref = item["reference_answer"]

            print(f"\n[{i+1}/{total}] 类别 {label}: {question}")
            print(f"正确答案: {ref}")

            llm_answer = self.get_answer(question).strip()
            print(f"LLM回答: {llm_answer}")

            results.append({
                "id": item["id"],
                "label": label,
                "question": question,
                "reference_answer": ref,
                "llm_answer": llm_answer,
                "user_score": None
            })

        return results

    def _scoring_phase(self, results):
        total = len(results)
        for i, r in enumerate(results):
            print(f"\n[{i+1}/{total}] 类别: {r['label']} | ID: {r['id']}")
            print(f"问题: {r['question']}")
            print(f"正确答案: {r['reference_answer']}")
            print(f"LLM回答: {r['llm_answer']}")
            while True:
                score = input("正确? (1=正确, 0=错误): ").strip()
                if score in ("0", "1"):
                    r["user_score"] = int(score)
                    break
                print("输入无效，请输入 0 或 1")

    def _save_results(self, results):
        os.makedirs(RESULT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(RESULT_DIR, f"benchmark_{timestamp}.json")

        total = len(results)
        correct = sum(1 for r in results if r["user_score"] == 1)
        acc = correct / total if total > 0 else 0

        output = {
            "accuracy": acc,
            "correct": correct,
            "total": total,
            "results": results
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"\n结果已保存到 {filepath}")
