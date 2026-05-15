import os
# hugging face镜像设置，如果国内环境无法使用启用该设置
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import torch

from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

load_dotenv()

markdown_path = "./data/C1/markdown/easy-rl-chapter1.md"

# 加载本地markdown文件
loader = UnstructuredMarkdownLoader(markdown_path)
docs = loader.load()

# 文本分块
text_splitter = RecursiveCharacterTextSplitter()
chunks = text_splitter.split_documents(docs)

# 中文嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="./models/Qwen3-Embedding-4B",   # 模型路径，可换成本地路径
    model_kwargs={'device': 'cpu'},         # 有GPU可改成 'cuda'
    encode_kwargs={'normalize_embeddings': True}  
)
  
# 构建向量存储
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(chunks)

# 提示词模板
prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”

上下文:
{context}

问题: {question}

回答:"""
                                          )

# 配置大语言模型

model_name_or_path = "./models/Qwen3.5-4B"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map="cpu",
    torch_dtype=torch.float32
)

# 用户查询
question = "文中举了哪些例子？"

# 在向量存储中查询相关文档
retrieved_docs = vectorstore.similarity_search(question, k=3)
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

print(f"上下文:{docs_content}")

del embeddings, vectorstore

torch.cuda.empty_cache()
messages = [
    {
        "role": "system",
        "content": "你是一个只能根据给定的上下文回答问题的助手。如果上下文没有答案，请明确告知用户。不要输出任何思考过程或解释。"
    },
    {
        "role": "user",
        "content": f"请根据下面提供的上下文信息来回答问题。\n\n上下文:\n{docs_content}\n\n问题: {question}"
    }
]
# messages = [
#     {"role": "user", "content": "你好，请用一句话介绍自己"}
# ]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True, 
    enable_thinking=False
)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **inputs,
    max_new_tokens=2048,
    do_sample=True,
    temperature=0.7
)

output_ids = generated_ids[0][len(inputs.input_ids[0]):]
response = tokenizer.decode(output_ids, skip_special_tokens=True)
print(response.strip())

# llm = ChatOpenAI(
#     model="deepseek-chat",
#     temperature=0.7,
#     max_tokens=4096,
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     base_url="https://api.deepseek.com"
# )