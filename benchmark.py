class Benchmark:
    def __init__(self, answer_func):
        """
        只传入一个函数：输入问题 → 返回回答字符串
        解耦核心！
        """
        self.get_answer = answer_func

    def run(self, benchmark_data):
        print("\n" + "=" * 60)
        print("📊 Benchmark 评测开始")
        print("=" * 60)

        total = len(benchmark_data)
        correct = 0

        for i, item in enumerate(benchmark_data):
            q = item["question"]
            ref = item["answer"]
            pred = self.get_answer(q).strip()

            print(f"\n[{i+1}/{total}] {q}")
            print(f"标准答案：{ref}")
            print(f"模型输出：{pred}")

            if ref in pred:
                print("✅ 正确")
                correct += 1
            else:
                print("❌ 错误")

        acc = correct / total
        print("\n" + "=" * 60)
        print(f"📊 评测结果：总题数 {total} | 正确 {correct} | 准确率 {acc:.2%}")
        print("=" * 60)
        return {"accuracy": acc, "correct": correct, "total": total}