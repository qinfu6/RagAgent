import os
import torch
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


markdown_path = "./doc/2/2025年学业指南.md"

# 加载文档
# loader = UnstructuredMarkdownLoader(markdown_path)
# docs = loader.load()
loader = TextLoader(markdown_path, encoding='utf-8')
docs = loader.load()

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

# 将检索步骤集成到链中
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": vectorstore.as_retriever() | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def get_answer(query):
    return rag_chain.invoke(query)


if __name__ == "__main__":
    question = "文中举了哪些例子？"
    answer = get_answer(question)
    print(f"答案: {answer.strip()}")

    from benchmark import Benchmark

    benchmark_data = [
        {"question": "文中举了哪些例子？", "answer": ""},
    ]

    bm = Benchmark(get_answer)
    bm.run(benchmark_data)