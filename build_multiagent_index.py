import os
import shutil
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- 1. 向量化模型初始化 ---
embeddings = HuggingFaceEmbeddings(
    model_name="./models/Qwen3-Embedding-4B",
    model_kwargs={'device': 'cpu'},  # 使用 CPU，显存不足使用 GPU 会导致结果出错
    encode_kwargs={'normalize_embeddings': True}
)

# --- 2. 向量库持久化路径 ---
persist_dir = "./faiss_index/multiagent"
data_dir = "./doc/5"

if os.path.exists(persist_dir) and os.listdir(persist_dir):
    answer = input(f"向量库已存在: {persist_dir}\n是否删除并重建? (y/n): ").strip().lower()
    if answer != 'y':
        print("已取消")
        exit(0)
    shutil.rmtree(persist_dir)
    print(f"已删除旧向量库: {persist_dir}")

os.makedirs(persist_dir, exist_ok=True)

# --- 3. 两阶段分块 ---
print(f"正在读取 {data_dir} 目录下的 Markdown 文件并进行两阶段切分...")

# Stage 1: 按标题层级切分
headers_to_split_on = [
    ("#", "Header_1"),
    ("##", "Header_2"),
    ("###", "Header_3"),
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

# Stage 2: 细切分
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80,
    length_function=len
)

final_chunks = []

for filepath in sorted(Path(data_dir).rglob("*.md")):
    file_name = filepath.name
    print(f"  处理: {file_name}")
    with open(filepath, "r", encoding="utf-8") as f:
        md_content = f.read()

    # 第一阶段：按标题切分成带有元数据的逻辑块
    macro_chunks = markdown_splitter.split_text(md_content)

    # 第二阶段：细切并继承标题元数据
    for m_chunk in macro_chunks:
        m_chunk.metadata["source"] = file_name
        micro_chunks = child_splitter.split_documents([m_chunk])
        final_chunks.extend(micro_chunks)

print(f"切分完成：共生成带有丰富结构元数据的小块 {len(final_chunks)} 个")

# --- 4. 构建 FAISS 并保存 ---
print("正在构建 FAISS 向量库...")
vectorstore = FAISS.from_documents(final_chunks, embeddings)
vectorstore.save_local(persist_dir)
print(f"向量库已创建并保存到 {persist_dir}")
