import os
import re
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

result_path = "./result/agent"

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


# --- 3. 准备生成模型 (使用 HuggingFacePipeline 包装) ---
model_name_or_path = "./models/Qwen3.5-4B"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map="cpu",  # 使用 CPU，显存不足使用 GPU 会导致结果出错
    torch_dtype=torch.float32
)

# 创建一个 transformers pipeline，并指定返回完整输出以便调试
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=False,
    temperature=0.1,
    return_full_text=False  # 只返回新生成的文本，不包含输入
)

# 使用 LangChain 的 HuggingFacePipeline 包装器
llm = HuggingFacePipeline(pipeline=pipe)

# --- 4. 构建 RAG 链 ---
# 提示词模板 (ChatPromptTemplate 会自动处理聊天格式)
system_prompt = """你是一个严格基于《学生手册》知识库的问答助手。你必须遵守以下铁律：

【绝对基于上下文】
1. 所有回答只能来自提供的【上下文】，不得使用任何外部知识或常识。
2. 如果上下文没有相关信息，必须直接回复：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”，禁止补充任何猜测、建议或解释。

【事实精准】
3. 答案中的数字、日期、专有名词必须与上下文原文完全一致，不得改写。
4. 答案末尾必须用括号注明依据来源，格式为：（参见第X条/第X页/某章节），如上下文无明确编号可写（依据原文）。

【忠实保留限制条件】
5. 必须完整保留原文中的所有限制词，如“仅限”“除……之外”“但是”“不得”“必须”“可以”等，严禁改变约束强度和语义方向。例如，不能将“可以”改为“必须”，不能将“不得”改为“原则上不建议”。

【复杂问题的证据链】
6. 当问题需要结合上下文中的多个条款或条件时，你必须先明确列出每一个相关条款的依据，再给出最终结论，确保逻辑桥接清晰，无关键证据遗漏。但总体回答仍应简洁，用“依据1: ...；依据2: ...；结论：...”的格式。
7. 禁止只凭局部信息直接给出结论，必须覆盖所有相关条件。

【输出格式】
- 先直接给出答案（一句话），然后换行后附上“依据：”和简要引用。
- 如果拒答，只需回复拒答语句，不得包含任何其他内容。"""

user_prompt = """上下文:
{context}

问题: {question}

回答:"""

# 构建提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", user_prompt)
])

# --- 决策层提示词：判断上下文能否回答问题 ---
decision_template = """你的任务是判断给定的上下文是否能够回答用户的问题。
如果上下文包含回答问题所需的信息，返回 1。
如果上下文不包含相关信息，返回 0。
只返回 0 或 1，不要返回任何其他内容。

上下文: {context}

问题: {question}

判断:"""
decision_prompt = ChatPromptTemplate.from_template(decision_template)

# --- Query2doc 提示词：生成伪文档 ---
# 用少样本提示让 LLM 生成一段与查询相关的伪文档
query2doc_template = """请根据问题生成一段可能包含答案和相关背景信息的文字。这段文字应该像一篇真实文档的片段，引入丰富的术语和事实，以帮助搜索引擎更好地匹配。
只返回生成的文档片段，不要包含其他内容。/no_think

示例1:
问题: 学校对于旷课的处理规定是什么？
文档片段: 根据学生手册第三十条，学生一学期内旷课累计达到10学时者，给予警告处分；达到20学时者，给予严重警告处分；达到30学时者，给予记过处分；达到40学时者，给予留校察看处分；达到50学时或以上者，给予开除学籍处分。旷课时间按实际授课时间计算，迟到、早退三次按旷课一学时计算。

示例2:
问题: 奖学金的评定标准有哪些？
文档片段: 奖学金评定主要依据学生的学业成绩、综合素质测评和思想品德表现。学业成绩要求本学年必修课无不及格科目，且平均学分绩点排名在专业前30%。综合素质测评包括科技创新、社会实践、文体活动等附加分。思想品德要求遵守校纪校规，无处分记录。具体等次和金额详见手册第15页。

现在，请根据以下问题生成文档片段：
问题: {question}
文档片段:"""
query2doc_prompt = ChatPromptTemplate.from_template(query2doc_template)

# 将检索步骤集成到链中
def format_docs(docs):
    blocks = []
    for doc in docs:
        h1 = doc.metadata.get("Header_1", "")
        h2 = doc.metadata.get("Header_2", "")
        h3 = doc.metadata.get("Header_3", "")
        source = doc.metadata.get("source", "")
        header_path = " -> ".join([h for h in [h1, h2, h3] if h])
        meta = f"【来源: {source} | 章节: {header_path}】" if header_path else f"【来源: {source}】"
        blocks.append(f"{meta}\n{doc.page_content}")
    context_str = "\n\n---\n\n".join(blocks)
    print("======== 检索到的上下文（含元数据） ========")
    print(context_str)
    print("==========================================\n")
    return context_str

retriever = vectorstore.as_retriever()

def clean_answer(text):
    print(f"原始text:{text}")
    text = re.sub(r'<think>[\s\S]*?</think>\s*', '', text)
    text = re.sub(r'</think>\s*', '', text)
    text = re.sub(r'^\s*Assistant:\s*', '', text)
    return text.strip()

def check_context_relevance(context, question):
    if not context.strip():
        return False
    chain = decision_prompt | llm | StrOutputParser()
    result = chain.invoke({"context": context, "question": question})
    result = clean_answer(result)
    print(f"[决策层agent]:{result}")
    return "1" in result.strip()

# ---------- 生成伪文档 ----------
def generate_pseudo_doc(question):
    """使用 Query2doc 方法生成伪文档"""
    messages = [
        {"role": "user", "content": query2doc_template.format(question=question)}
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False        # 核心参数：禁止生成思考过程
    )
    raw_pseudo_doc = llm.invoke(prompt_text)
    pseudo_doc = clean_answer(raw_pseudo_doc)
    # chain = query2doc_prompt | llm | StrOutputParser()
    # pseudo_doc = chain.invoke({"question": question})
    # pseudo_doc = clean_answer(pseudo_doc)
    print(f"[Query2doc 伪文档]:\n{pseudo_doc}\n")
    return pseudo_doc

def expand_query_with_pseudo_doc(original_query, pseudo_doc, max_length=400):
    """将原始查询与伪文档拼接，形成扩展查询，并限制长度以适应嵌入模型"""
    expanded = f"{original_query} {pseudo_doc}"
    # 简单截断，也可用 tokenizer 精确截断
    if len(expanded) > max_length:
        expanded = expanded[:max_length]
    return expanded


def get_answer(query):
    docs = retriever.invoke(query)
    local_context = format_docs(docs)

    if check_context_relevance(local_context, query):
        print("决策层通过，进行最终回答")
        answer_context = local_context
    else:
        print("决策层未通过，使用 Query2doc 进行查询扩展并二次检索")
        # 生成伪文档
        pseudo_doc = generate_pseudo_doc(query)
        # 拼接原始查询与伪文档
        expanded_query = expand_query_with_pseudo_doc(query, pseudo_doc)
        print(f"[扩展查询]: {expanded_query[:200]}...")  # 仅打印前部分
        # 第二次检索
        docs2 = retriever.invoke(expanded_query)
        answer_context = format_docs(docs2)

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": answer_context, "question": query}).strip()
    answer = clean_answer(answer)
    return answer, answer_context


if __name__ == "__main__":
    bm = Benchmark(get_answer, result_path)
    bm.run()