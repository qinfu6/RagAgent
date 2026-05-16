import os
import re
import torch
from pathlib import Path
import requests
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

# --- 1. 多数据源加载：扫描指定文件夹下所有 .md 文件 ---
def load_markdown_files(data_dir):
    all_docs = []
    for filepath in sorted(Path(data_dir).rglob("*.md")):
        loader = TextLoader(str(filepath), encoding='utf-8')
        all_docs.extend(loader.load())
    return all_docs

data_dir = "./doc/3"
docs = load_markdown_files(data_dir)
print(f"已加载 {len(docs)} 个文档")

# 文本分块（更新方法为 create_documents）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)
chunks = text_splitter.split_documents(docs)

# 向量化与存储
embeddings = HuggingFaceEmbeddings(
    model_name="./models/Qwen3-Embedding-4B",
    model_kwargs={'device': 'cpu'},  # 使用 CPU，显存不足使用 GPU 会导致结果出错
    encode_kwargs={'normalize_embeddings': True}
)

vectorstore = FAISS.from_documents(chunks, embeddings)


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
template = """请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知："抱歉，我无法根据提供的上下文找到相关信息来回答此问题。"

上下文:
{context}

问题: {question}

回答:"""
prompt = ChatPromptTemplate.from_template(template)

# --- 决策层提示词：判断上下文能否回答问题 ---
decision_template = """你的任务是判断给定的上下文是否能够回答用户的问题。
如果上下文包含回答问题所需的信息，返回 1。
如果上下文不包含相关信息，返回 0。
只返回 0 或 1，不要返回任何其他内容。

上下文: {context}

问题: {question}

判断:"""
decision_prompt = ChatPromptTemplate.from_template(decision_template)

# 将检索步骤集成到链中
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

retriever = vectorstore.as_retriever()

def check_context_relevance(context, question):
    if not context.strip():
        return False
    chain = decision_prompt | llm | StrOutputParser()
    result = chain.invoke({"context": context, "question": question})
    return "1" in result.strip()

def web_search(query, max_results=5):
    try:
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=10
        )
        data = resp.json()
        parts = []
        if data.get("AbstractText"):
            parts.append(data["AbstractText"])
        for topic in data.get("RelatedTopics", []):
            if isinstance(topic, dict) and topic.get("Text"):
                parts.append(topic["Text"])
                if len(parts) >= max_results:
                    break
        if parts:
            return "\n\n".join(parts)
    except Exception:
        pass
    return ""

def get_answer(query):
    docs = retriever.invoke(query)
    local_context = format_docs(docs)

    if check_context_relevance(local_context, query):
        answer_context = local_context
    else:
        web_context = web_search(query)
        if web_context:
            answer_context = web_context
        else:
            answer_context = local_context

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": answer_context, "question": query}).strip()


if __name__ == "__main__":
    bm = Benchmark(get_answer)
    bm.run()