import os
import re
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# md loader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline # 正确方式：从独立包导入
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from benchmark import Benchmark

result_path = "./result/multi-agent"

# --- 1. 多数据源加载：扫描指定文件夹下所有 .md 文件 ---
def load_markdown_files(data_dir):
    all_docs = []
    for filepath in sorted(Path(data_dir).rglob("*.md")):
        loader = TextLoader(str(filepath), encoding='utf-8')
        all_docs.extend(loader.load())
    return all_docs

# 向量化
embeddings = HuggingFaceEmbeddings(
    model_name="./models/Qwen3-Embedding-4B",
    model_kwargs={'device': 'cpu'},  # 使用 CPU，显存不足使用 GPU 会导致结果出错
    encode_kwargs={'normalize_embeddings': True}
)

# FAISS 持久化：如果向量库已存在则直接加载, 否则创建并保存
persist_dir = "./faiss_index/agent"
if os.path.exists(persist_dir) and os.listdir(persist_dir):
    vectorstore = FAISS.load_local(
        persist_dir, embeddings, allow_dangerous_deserialization=True
    )
    print("已加载已有向量库")
else:
    os.makedirs(persist_dir, exist_ok=True)
    data_dir = "./doc/3"
    docs = load_markdown_files(data_dir)
    print(f"已加载 {len(docs)} 个文档")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(persist_dir)
    print("向量库已创建并保存")


# --- 3. 加载基础模型 (所有 Agent 共享 tokenizer + model) ---
model_name_or_path = "./models/Qwen3.5-4B"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map="cpu",  # 使用 CPU，显存不足使用 GPU 会导致结果出错
    torch_dtype=torch.float32
)

# --- 4. 多 Agent 架构 ---

class VerifiedResult:
    def __init__(self, valid, reason):
        self.valid = valid
        self.reason = reason


class BaseAgent:
    def __init__(self, model, tokenizer, max_new_tokens, temperature, do_sample):
        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            return_full_text=False
        )
        self.llm = HuggingFacePipeline(pipeline=self.pipe)

    @staticmethod
    def clean_answer(text):
        text = re.sub(r'<think>[\s\S]*?</think>\s*', '', text)
        text = re.sub(r'</think>\s*', '', text)
        text = re.sub(r'^\s*Assistant:\s*', '', text)
        return text.strip()

    @staticmethod
    def _to_chat_messages(langchain_messages):
        role_map = {"system": "system", "human": "user", "ai": "assistant"}
        return [{"role": role_map.get(msg.type, msg.type), "content": msg.content} for msg in langchain_messages]

    def invoke_llm(self, messages):
        if messages and isinstance(messages[0], dict):
            dict_messages = messages
        else:
            dict_messages = self._to_chat_messages(messages)
        prompt_text = self.pipe.tokenizer.apply_chat_template(
            dict_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        raw = self.llm.invoke(prompt_text)
        return self.clean_answer(raw)


# ==================== RetrieverAgent ====================
class RetrieverAgent:
    def __init__(self, vectorstore, k=3, fetch_k=10, lambda_mult=0.6):
        self.vectorstore = vectorstore
        self.k = k
        self.fetch_k = fetch_k          # 候选池大小
        self.lambda_mult = lambda_mult  # 多样性控制

    def search(self, query):
        # docs = self.vectorstore.similarity_search(query, k=self.k)
        # 使用 MMR 检索，返回既相关又多样的文档
        docs = self.vectorstore.max_marginal_relevance_search(
            query,
            k=self.k,
            fetch_k=self.fetch_k,
            lambda_mult=self.lambda_mult
        )
        context = "\n\n".join(doc.page_content for doc in docs)
        print("======== 检索到的上下文 ========")
        print(context)
        print("================================\n")
        return context


# ==================== DecisionAgent ====================
class DecisionAgent(BaseAgent):
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer, max_new_tokens=10, temperature=0, do_sample=False)
        self.prompt = ChatPromptTemplate.from_template("""你的任务是判断给定的上下文是否能够回答用户的问题。
如果上下文包含回答问题所需的信息，返回 1。
如果上下文不包含相关信息，返回 0。
只返回 0 或 1，不要返回任何其他内容。

上下文: {context}

问题: {question}

判断:""")

    def check(self, context, question):
        if not context.strip():
            return False
        messages = self.prompt.format_messages(context=context, question=question)
        result = self.invoke_llm(messages)
        print(f"[DecisionAgent]: {result.strip()}")
        return "1" in result.strip()


# ==================== ReformulatorAgent (已弃用, 保留供参考) ====================
# class ReformulatorAgent(BaseAgent):
#     def __init__(self, model, tokenizer):
#         super().__init__(model, tokenizer, max_new_tokens=64, temperature=0.3, do_sample=True)
#         self.prompt = ChatPromptTemplate.from_template("""你是一个搜索查询优化助手。
# 请根据原始问题，提取核心关键词或改写为更适合在文档库中检索的表述。
# 只返回优化后的检索词，不要返回任何其他内容。
#
# 原始问题: {question}
#
# 优化检索词:""")
#
#     def rewrite(self, question):
#         messages = self.prompt.format_messages(question=question)
#         result = self.invoke_llm(messages)
#         rewritten = result.strip()
#         print(f"[ReformulatorAgent]: {rewritten}")
#         return rewritten


def expand_query_with_pseudo_doc(original_query, pseudo_doc, max_length=400):
    expanded = f"{original_query} {pseudo_doc}"
    if len(expanded) > max_length:
        expanded = expanded[:max_length]
    return expanded


# ==================== Query2docAgent ====================
class Query2docAgent(BaseAgent):
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer, max_new_tokens=256, temperature=0.3, do_sample=True)
        self.few_shot_prompt = """请根据问题生成一段可能包含答案和相关背景信息的文字。这段文字应该像一篇真实文档的片段，引入丰富的术语和事实，以帮助搜索引擎更好地匹配。
只返回生成的文档片段，不要包含其他内容。

示例1:
问题: 学校对于旷课的处理规定是什么？
文档片段: 根据学生手册第三十条，学生一学期内旷课累计达到10学时者，给予警告处分；达到20学时者，给予严重警告处分；达到30学时者，给予记过处分；达到40学时者，给予留校察看处分；达到50学时或以上者，给予开除学籍处分。旷课时间按实际授课时间计算，迟到、早退三次按旷课一学时计算。

示例2:
问题: 奖学金的评定标准有哪些？
文档片段: 奖学金评定主要依据学生的学业成绩、综合素质测评和思想品德表现。学业成绩要求本学年必修课无不及格科目，且平均学分绩点排名在专业前30%。综合素质测评包括科技创新、社会实践、文体活动等附加分。思想品德要求遵守校纪校规，无处分记录。具体等次和金额详见手册第15页。

现在，请根据以下问题生成文档片段：
问题: {question}
文档片段:"""

    def generate(self, question):
        messages = [{"role": "user", "content": self.few_shot_prompt.format(question=question)}]
        return self.invoke_llm(messages)


# ==================== AnswerAgent ====================
class AnswerAgent(BaseAgent):
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer, max_new_tokens=512, temperature=0.1, do_sample=False)
        self.system_prompt = """你是一个严格基于《学生手册》知识库的问答助手。你必须遵守以下铁律：

【绝对基于上下文】
1. 所有回答只能来自提供的【上下文】，不得使用任何外部知识或常识。
2. 如果上下文没有相关信息，必须直接回复："抱歉，我无法根据提供的上下文找到相关信息来回答此问题。"，禁止补充任何猜测、建议或解释。

【事实精准】
3. 答案中的数字、日期、专有名词必须与上下文原文完全一致，不得改写。
4. 答案末尾必须用括号注明依据来源，格式为：（参见第X条/第X页/某章节），如上下文无明确编号可写（依据原文）。

【忠实保留限制条件】
5. 必须完整保留原文中的所有限制词，如"仅限""除……之外""但是""不得""必须""可以"等，严禁改变约束强度和语义方向。例如，不能将"可以"改为"必须"，不能将"不得"改为"原则上不建议"。

【复杂问题的证据链】
6. 当问题需要结合上下文中的多个条款或条件时，你必须先明确列出每一个相关条款的依据，再给出最终结论，确保逻辑桥接清晰，无关键证据遗漏。但总体回答仍应简洁，用"依据1: ...；依据2: ...；结论：..."的格式。
7. 禁止只凭局部信息直接给出结论，必须覆盖所有相关条件。

【输出格式】
- 先直接给出答案（一句话），然后换行后附上"依据："和简要引用。
- 如果拒答，只需回复拒答语句，不得包含任何其他内容。"""

        self.base_user_template = """上下文:
{context}

问题: {question}

回答:"""

    def _build_messages(self, context, question):
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.base_user_template.format(context=context, question=question)}
        ]

    def generate(self, context, question):
        messages = self._build_messages(context, question)
        return self.invoke_llm(messages)

    def regenerate_with_feedback(self, context, question, previous_answer, feedback_reason):
        messages = self._build_messages(context, question)
        messages.append({"role": "assistant", "content": previous_answer})
        feedback_msg = (
            f"你的上一个回答没有通过事实核查，原因如下：{feedback_reason}\n"
            f"请严格基于提供的上下文重新回答，纠正上述问题。如果上下文确实无法支持，请明确拒答。"
        )
        messages.append({"role": "user", "content": feedback_msg})
        return self.invoke_llm(messages)


# ==================== VerifierAgent ====================
class VerifierAgent(BaseAgent):
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer, max_new_tokens=128, temperature=0, do_sample=False)
        self.prompt = ChatPromptTemplate.from_template("""你是学生手册问答的严格审查员。请验证以下回答是否完全忠于提供的上下文。

验证标准：
1. 回答中的每一条事实能否在上下文中找到原文一字不差的支撑？
2. 回答是否编造了上下文中不存在的信息？
3. 回答是否正确保留了原文的限制词（可以/必须/不得/仅限等），没有改变约束强度？

请用 JSON 格式返回，不要返回其他内容：
{{"valid": true/false, "reason": "如果不合法，说明具体问题"}}

上下文: {context}

问题: {question}

待验证回答: {answer}

审查结果 JSON:""")

    def verify(self, context, answer, question):
        messages = self.prompt.format_messages(context=context, answer=answer, question=question)
        result = self.invoke_llm(messages)
        print(f"[VerifierAgent]: {result.strip()}")

        try:
            data = json.loads(result.strip())
            return VerifiedResult(valid=data.get("valid", False), reason=data.get("reason", "未知错误"))
        except json.JSONDecodeError:
            valid = re.search(r'"valid"\s*:\s*(true|false)', result, re.IGNORECASE)
            reason = re.search(r'"reason"\s*:\s*"([^"]*)"', result)
            return VerifiedResult(
                valid=(valid and valid.group(1).lower() == "true"),
                reason=reason.group(1) if reason else "无法解析验证结果"
            )


# ==================== OrchestratorAgent ====================
class OrchestratorAgent:
    def __init__(self, retriever, decision, query2doc, answer, verifier, max_retries=2):
        self.retriever = retriever
        self.decision = decision
        self.query2doc = query2doc
        self.answer = answer
        self.verifier = verifier
        self.max_retries = max_retries

    def run(self, query):
        context = None

        for round_idx in range(self.max_retries + 1):
            if round_idx == 0:
                search_query = query
                print(f"\n[Orchestrator] 第 {round_idx + 1} 轮检索，原始查询")
            else:
                pseudo_doc = self.query2doc.generate(query)
                search_query = expand_query_with_pseudo_doc(query, pseudo_doc)
                print(f"\n[Orchestrator] 第 {round_idx + 1} 轮检索，Query2doc 扩展查询")

            context = self.retriever.search(search_query)
            if self.decision.check(context, query):
                print(f"[Orchestrator] 决策通过，上下文可用")
                break
        else:
            print(f"[Orchestrator] 全部 {self.max_retries + 1} 轮检索未通过，使用最后一轮上下文")

        answer = self.answer.generate(context, query)

        for verify_round in range(2):
            result = self.verifier.verify(context, answer, query)
            if result.valid:
                print(f"[Orchestrator] 验证通过 (第 {verify_round + 1} 次)")
                break
            print(f"[Orchestrator] 验证不通过: {result.reason}")
            answer = self.answer.regenerate_with_feedback(
                context=context,
                question=query,
                previous_answer=answer,
                feedback_reason=result.reason
            )
        else:
            print("[Orchestrator] 二次验证仍不通过，返回最终回答")

        return answer, context


# --- 5. 实例化各 Agent ---
retriever_agent = RetrieverAgent(vectorstore, k=3)
decision_agent = DecisionAgent(model, tokenizer)
# reformulator_agent = ReformulatorAgent(model, tokenizer)  # 已弃用
query2doc_agent = Query2docAgent(model, tokenizer)
answer_agent = AnswerAgent(model, tokenizer)
verifier_agent = VerifierAgent(model, tokenizer)

orchestrator = OrchestratorAgent(
    retriever=retriever_agent,
    decision=decision_agent,
    query2doc=query2doc_agent,
    answer=answer_agent,
    verifier=verifier_agent,
    max_retries=2
)

def get_answer(query):
    return orchestrator.run(query)


if __name__ == "__main__":
    bm = Benchmark(get_answer, result_path)
    bm.run()