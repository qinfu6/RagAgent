import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

embedding = HuggingFaceEmbeddings(
    model_name="./models/Qwen3-Embedding-4B",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)

tokenizer = AutoTokenizer.from_pretrained("./models/Qwen3.5-4B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "./models/Qwen3.5-4B",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.1,
    top_p=0.95,
    do_sample=False
)
llm = HuggingFacePipeline(pipeline=pipe)

loader = PyPDFLoader("2025学生手册.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50
)
splits = text_splitter.split_documents(docs)

db = FAISS.from_documents(splits, embedding)
retriever = db.as_retriever(search_kwargs={"k": 3})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    get_source_documents=True
)

def chat(query):
    res = qa_chain.invoke({"query": query})
    print("问题：", res["query"])
    print("回答：", res["result"])
    print("来源片段：")
    for doc in res["source_documents"]:
        print("-", doc.page_content[:100] + "...")

if __name__ == "__main__":
    chat("缺勤几次会扣分？")