import os
import re
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

result_path = "./result/baseline"

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
persist_dir = "./faiss_index/baseline"
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
    max_new_tokens=256,
    do_sample=False,
    temperature=0.1,
    return_full_text=False  # 只返回新生成的文本，不包含输入
)

# 使用 LangChain 的 HuggingFacePipeline 包装器
llm = HuggingFacePipeline(pipeline=pipe)

# --- 4. 构建 RAG 链 ---
# 提示词模板
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
- 如果拒答，只需回复拒答语句，不得包含任何其他内容。/no_think"""

user_prompt = """上下文:
{context}

问题: {question}

回答:"""

# 构建提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", user_prompt)
])

# 将检索步骤集成到链中
def format_docs(docs):
    context_str = "\n\n".join(doc.page_content for doc in docs)
    # 打印检索到的上下文
    print("======== 检索到的上下文 ========")
    print(context_str)
    print("================================\n")
    return context_str

retriever = vectorstore.as_retriever()

def clean_answer(text):
    text = re.sub(r'<think>[\s\S]*?</think>\s*', '', text)
    text = re.sub(r'</think>\s*', '', text)
    text = re.sub(r'^\s*Assistant:\s*', '', text)
    return text.strip()

def get_answer(query):
    docs = retriever.invoke(query)
    context = format_docs(docs)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": query}).strip()
    answer = clean_answer(answer)
    return answer, context


if __name__ == "__main__":
    bm = Benchmark(get_answer, result_path)
    bm.run()