import os
import re
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# md loader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
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
persist_dir = "./faiss_index/multiagent"
if os.path.exists(persist_dir) and os.listdir(persist_dir):
    vectorstore = FAISS.load_local(
        persist_dir, embeddings, allow_dangerous_deserialization=True
    )
    print("已加载已有向量库")
else:
    os.makedirs(persist_dir, exist_ok=True)
    data_dir = "./doc/5"
    print(f"正在读取 {data_dir} 目录下的 Markdown 文件并进行两阶段切分...")

    headers_to_split_on = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        length_function=len
    )

    final_chunks = []
    for filepath in sorted(Path(data_dir).rglob("*.md")):
        file_name = filepath.name
        with open(filepath, "r", encoding="utf-8") as f:
            md_content = f.read()
        macro_chunks = markdown_splitter.split_text(md_content)
        for m_chunk in macro_chunks:
            m_chunk.metadata["source"] = file_name
            micro_chunks = child_splitter.split_documents([m_chunk])
            final_chunks.extend(micro_chunks)

    print(f"切分完成：共生成带有丰富结构元数据的小块 {len(final_chunks)} 个")
    vectorstore = FAISS.from_documents(final_chunks, embeddings)
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
            max_length=None,  # 消除与 max_new_tokens 的冲突警告
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
        blocks = []
        for doc in docs:
            h1 = doc.metadata.get("Header_1", "")
            h2 = doc.metadata.get("Header_2", "")
            h3 = doc.metadata.get("Header_3", "")
            source = doc.metadata.get("source", "")
            header_path = " -> ".join([h for h in [h1, h2, h3] if h])
            meta = f"【来源: {source} | 章节: {header_path}】" if header_path else f"【来源: {source}】"
            blocks.append(f"{meta}\n{doc.page_content}")
        context = "\n\n---\n\n".join(blocks)
        print("======== 检索到的上下文（含元数据） ========")
        print(context)
        print("==========================================\n")
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
            f"你的上一个回答不够完整，原因如下：{feedback_reason}\n"
            f"请基于提供的上下文补充上述遗漏信息后重新回答。只要上下文中有相关信息，就一定不要拒绝回答。如果上下文确实完全无法支持，再明确拒答。"
        )
        messages.append({"role": "user", "content": feedback_msg})
        return self.invoke_llm(messages)


# ==================== VerifierAgent ====================
class VerifierAgent(BaseAgent):
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer, max_new_tokens=512, temperature=0, do_sample=False)  # VerifierAgent
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

        # 尝试解析 JSON
        stripped = result.strip()
        try:
            data = json.loads(stripped)
            return VerifiedResult(valid=data.get("valid", False), reason=data.get("reason", "未知错误"))
        except json.JSONDecodeError:
            pass

        # 截断修复：reason 字符串中途截断时补全 JSON
        fixed = re.sub(r'"[^"]*$', '"}', stripped)
        try:
            data = json.loads(fixed)
            print("[VerifierAgent] JSON 已自动修补（可能被截断）")
            return VerifiedResult(valid=data.get("valid", False), reason=data.get("reason", "解析截断，部分内容丢失"))
        except json.JSONDecodeError:
            pass

        # 最后兜底：regex 提取，标记低可信度
        valid = re.search(r'"valid"\s*:\s*(true|false)', result, re.IGNORECASE)
        reason = re.search(r'"reason"\s*:\s*"([^"]*)"', result)
        return VerifiedResult(
            valid=(valid and valid.group(1).lower() == "true"),
            reason=(reason.group(1) if reason else f"[解析失败] 原始输出: {stripped[:60]}...")
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


# ==================== ReActAgent ====================
class ReActAgent(BaseAgent):
    def __init__(self, model, tokenizer, retriever, answer):
        super().__init__(model, tokenizer, max_new_tokens=1024, temperature=0.1, do_sample=False)
        self.retriever = retriever
        self.answer = answer
        self.max_iterations = 5

        self.system_prompt = """你是一个基于《学生手册》知识库的问答助手。你需要通过"推理-行动"循环逐步收集信息来回答问题。

可用工具：
- Search[查询词]：在文档库中搜索相关信息。你应该从不同角度多次搜索以获取完整信息。
- Finish：当你认为已收集到足够信息时使用此行动。

严格按以下格式输出：
Thought: <你当前的分析和思考>
Action: Search[<具体的查询词>]

例如：
当认为需要搜索相关信息时：
Thought: <你的分析和思考>
Action: Search[<具体的查询词>]

当认为信息足够时：
Thought: <总结你收集到的信息>
Action: Finish

搜寻结束后，基于收集到的所有上下文给出最终答案。"""

    def _parse_react_output(self, text):
        thought_match = re.search(r'Thought:\s*(.*?)(?=\n*(?:Action:|$))', text, re.DOTALL | re.IGNORECASE)
        action_search = re.search(r'Action:\s*Search\s*\[\s*(.*?)\s*\]', text, re.IGNORECASE)
        action_finish = re.search(r'Action:\s*Finish', text, re.IGNORECASE)
        return {
            "thought": thought_match.group(1).strip() if thought_match else "",
            "action_search": action_search,
            "action_finish": action_finish,
        }

    def run(self, query):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query}
        ]

        context_parts = []

        for iteration in range(1, self.max_iterations + 1):
            print(f"\n[ReActAgent] 第 {iteration} 轮推理")
            response = self.invoke_llm(messages)
            print(f"[ReActAgent] LLM 输出:\n{response}")

            parsed = self._parse_react_output(response)

            messages.append({"role": "assistant", "content": response})

            if parsed["action_search"]:
                search_query = parsed["action_search"].group(1).strip()
                print(f"[ReActAgent] 执行搜索: {search_query}")
                observation = self.retriever.search(search_query)
                context_parts.append(observation)
                messages.append({"role": "user", "content": f"Observation:\n{observation}"})
            elif parsed["action_finish"]:
                print(f"[ReActAgent] LLM 决定 Finish，循环结束")
                break
            else:
                print(f"[ReActAgent] 无法解析行动，循环结束")
                break
        else:
            print(f"[ReActAgent] 达到最大迭代次数 {self.max_iterations}")

        accumulated_context = "\n\n".join(context_parts) if context_parts else ""
        if accumulated_context:
            print(f"[ReActAgent] 精炼最终答案...")
            answer = self.answer.generate(accumulated_context, query)
        else:
            answer = "抱歉，我无法根据提供的上下文找到相关信息来回答此问题。"

        return answer, accumulated_context


# ==================== ReActOrchestratorAgent (ReAct + Multi-Agent 管道) ====================
class ReActOrchestratorAgent(BaseAgent):
    def __init__(self, model, tokenizer, retriever, decision, query2doc, answer, verifier):
        super().__init__(model, tokenizer, max_new_tokens=1024, temperature=0.1, do_sample=False)
        self.retriever = retriever
        self.decision = decision
        self.query2doc = query2doc
        self.answer = answer
        self.verifier = verifier
        self.max_iterations = 5

        self.system_prompt = """你是一个基于《学生手册》知识库的问答助手。你需要通过"推理-行动"循环逐步收集信息来回答问题。

可用工具：
- Search[查询词]：在文档库中搜索相关信息。你应该从不同角度多次搜索以获取完整信息。
- Finish：当你认为已收集到足够信息时使用此行动。

严格按以下格式输出：

Thought: <你当前的分析和思考>
Action: Search[<具体的查询词>]

当认为信息足够时：

Thought: <总结你收集到的信息>
Action: Finish

搜寻结束后，基于收集到的所有上下文给出最终答案。"""

    def _parse_react_output(self, text):
        thought_match = re.search(r'Thought:\s*(.*?)(?=\n*(?:Action:|$))', text, re.DOTALL | re.IGNORECASE)
        action_search = re.search(r'Action:\s*Search\s*\[\s*(.*?)\s*\]', text, re.IGNORECASE)
        action_finish = re.search(r'Action:\s*Finish', text, re.IGNORECASE)
        return {
            "thought": thought_match.group(1).strip() if thought_match else "",
            "action_search": action_search,
            "action_finish": action_finish,
        }

    def run(self, query):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query}
        ]

        context_parts = []
        last_reasoning = None

        # ── Phase 1: ReAct 自主搜索（含 DecisionAgent 智能提示） ──
        for iteration in range(1, self.max_iterations + 1):
            print(f"\n[ReActOrchestrator] 第 {iteration} 轮推理")
            response = self.invoke_llm(messages)
            print(f"[ReActOrchestrator] LLM 输出:\n{response}")

            parsed = self._parse_react_output(response)
            messages.append({"role": "assistant", "content": response})

            if parsed["action_search"]:
                search_query = parsed["action_search"].group(1).strip()
                # Query2doc：生成伪文档扩展查询，提升检索精度
                pseudo_doc = self.query2doc.generate(search_query)
                expanded_query = expand_query_with_pseudo_doc(search_query, pseudo_doc)
                print(f"[ReActOrchestrator] Query2doc 扩展: {search_query} → {expanded_query[:100]}...")
                print(f"[ReActOrchestrator] 执行搜索: {expanded_query}")
                observation = self.retriever.search(expanded_query)
                context_parts.append(observation)

                # DecisionAgent 静默检查：累积上下文是否已足够
                accumulated = "\n\n".join(context_parts)
                if self.decision.check(accumulated, query):
                    observation += "\n\n[系统提示: 当前已收集到足够信息，可以考虑 Finish]"

                messages.append({"role": "user", "content": f"Observation:\n{observation}"})
            elif parsed["action_finish"]:
                print(f"[ReActOrchestrator] LLM 决定 Finish，循环结束")
                last_reasoning = parsed["thought"]
                break
            else:
                print(f"[ReActOrchestrator] 无法解析行动，循环结束")
                break
        else:
            print(f"[ReActOrchestrator] 达到最大迭代次数 {self.max_iterations}")

        # ── Phase 2: AnswerAgent 精炼 + VerifierAgent 验证修正 ──
        accumulated_context = "\n\n".join(context_parts) if context_parts else ""
        if last_reasoning:
            accumulated_context += f"\n\n[推理摘要]\n{last_reasoning}"
        if accumulated_context:
            print(f"[ReActOrchestrator] 精炼最终答案并验证...")
            answer = self.answer.generate(accumulated_context, query)

            for verify_round in range(2):
                result = self.verifier.verify(accumulated_context, answer, query)
                if result.valid:
                    print(f"[ReActOrchestrator] 验证通过 (第 {verify_round + 1} 次)")
                    break
                print(f"[ReActOrchestrator] 验证不通过: {result.reason}")
                answer = self.answer.regenerate_with_feedback(
                    context=accumulated_context,
                    question=query,
                    previous_answer=answer,
                    feedback_reason=result.reason
                )
            else:
                print("[ReActOrchestrator] 二次验证仍不通过，返回最终回答")
        else:
            answer = "抱歉，我无法根据提供的上下文找到相关信息来回答此问题。"

        return answer, accumulated_context


# --- 5. 实例化各 Agent ---
retriever_agent = RetrieverAgent(vectorstore, k=3)
decision_agent = DecisionAgent(model, tokenizer)
# reformulator_agent = ReformulatorAgent(model, tokenizer)  # 已弃用
query2doc_agent = Query2docAgent(model, tokenizer)
answer_agent = AnswerAgent(model, tokenizer)
verifier_agent = VerifierAgent(model, tokenizer)

ORCHESTRATOR_MODE = "react_orch"  # "orch" | "react" | "react_orch"

if ORCHESTRATOR_MODE == "orch":
    orchestrator = OrchestratorAgent(
        retriever=retriever_agent,
        decision=decision_agent,
        query2doc=query2doc_agent,
        answer=answer_agent,
        verifier=verifier_agent,
        max_retries=2
    )
elif ORCHESTRATOR_MODE == "react":
    orchestrator = ReActAgent(model, tokenizer, retriever=retriever_agent, answer=answer_agent)
elif ORCHESTRATOR_MODE == "react_orch":
    orchestrator = ReActOrchestratorAgent(
        model, tokenizer,
        retriever=retriever_agent,
        decision=decision_agent,
        query2doc=query2doc_agent,
        answer=answer_agent,
        verifier=verifier_agent
    )

def get_answer(query):
    return orchestrator.run(query)


if __name__ == "__main__":
    bm = Benchmark(get_answer, result_path)
    bm.run()