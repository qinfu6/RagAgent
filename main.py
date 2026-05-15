import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===================== 1. 加载 Qwen3-Embedding =====================
embedding = HuggingFaceEmbeddings(
    model_name="./Qwen3-Embedding-4B",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)

# ===================== 2. 加载大模型 =====================
print("✅ 加载大模型中...")
tokenizer = AutoTokenizer.from_pretrained("./Qwen3.5-4B")
model = AutoModelForCausalLM.from_pretrained(
    "./Qwen3.5-4B",
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True
)

# ===================== 3. 读取并切分 PDF =====================
print("✅ 读取PDF中...")
loader = PyPDFLoader("2025学生手册.pdf")
docs = loader.load()

print("✅ 文本分块中...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30
)
splits = text_splitter.split_documents(docs)

# ===================== 4. 构建向量库 =====================
print("✅ 构建向量库中...")
db = FAISS.from_documents(splits, embedding)
retriever = db.as_retriever(search_kwargs={"k": 2})

# ===================== 核心函数：给外部调用（Benchmark / Agent） =====================
def get_answer(query):
    docs = retriever.invoke(query)
    context = "\n".join([d.page_content for d in docs])

    prompt = f"""请根据以下资料回答问题，不要编造。
资料：{context}
问题：{query}
回答："""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=False
    )
    answer = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    return answer

# ===================== 聊天函数 =====================
def chat(query):
    print("\n🔍 问题：", query)
    answer = get_answer(query)
    print("💡 回答：", answer)

# ===================== 运行 + 解耦 BENCHMARK =====================
if __name__ == "__main__":
    print("✅ 系统启动成功！")

    # 测试对话
    chat("学生缺勤几次会扣分？")

    # ===================== 解耦版 Benchmark 接入 =====================
    from benchmark import Benchmark

    benchmark_data = [
        {"question": "学生缺勤几次会扣分？", "answer": "一学期缺勤累计10次会扣平时分"},
        {"question": "迟到多久算缺勤？", "answer": "迟到15分钟及以上记为缺勤"},
        {"question": "学生旷课会受到什么处分？", "answer": "旷课累计15节以上给予警告处分"},
    ]

    bm = Benchmark(get_answer)
    bm.run(benchmark_data)