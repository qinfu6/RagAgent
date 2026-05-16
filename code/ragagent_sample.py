import os
import PyPDF2
from tqdm.notebook import tqdm
import re
import json

from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from langchain_openai import AzureOpenAIEmbeddings
# from langchain.chat_models import AzureChatOpenAI
from openai import AzureOpenAI

load_dotenv()

# --- Read and load PDF files ---

ES_USER= os.getenv("ES_USER")
ES_PASSWORD = os.getenv("ES_PASSWORD")
ES_ENDPOINT = os.getenv("ES_ENDPOINT")

MODEL_NAME = os.getenv("MODEL_NAME")
AZURE_EMBEDDING_ENDPOINT = os.getenv("AZURE_EMBEDDING_ENDPOINT")
AZURE_EMBEDDING_API_KEY = os.getenv("AZURE_EMBEDDING_API_KEY")
AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_EMBEDDING_API_VERSION")

AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_EDNPOINT = os.getenv("AZURE_EDNPOINT")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
AZURE_DEPLOYMENT_ID = os.getenv("AZURE_DEPLOYMENT_ID")

url = f"https://{ES_USER}:{ES_PASSWORD}@{ES_ENDPOINT}:9200"
es = Elasticsearch(url, ca_certs = "./http_ca.crt", verify_certs = True)

print(es.info())

elastic_index_name = "agent_rag_index"

embeddings = AzureOpenAIEmbeddings(
    model=MODEL_NAME,
    azure_endpoint=AZURE_EMBEDDING_ENDPOINT, 
    api_key= AZURE_EMBEDDING_API_KEY,
    openai_api_version=AZURE_EMBEDDING_API_VERSION
)

chat = AzureOpenAI(
  api_key = AZURE_API_KEY,  
  api_version = AZURE_API_VERSION,
  azure_endpoint = AZURE_EDNPOINT
)


def read_pdfs_from_folder(folder_path):
    pdf_list = []
    
    # Loop through all files in the specified folder
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            
            # Open each PDF file
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                content = ""
                
                # Read each page's content and append it to a string
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    content += page.extract_text()
                
                # Add the PDF content to the list
                pdf_list.append({"content": content, "filename": filename})
    
    return pdf_list

folder_path = "./rag_data"

# all_documents = read_pdfs_from_folder(folder_path)

# --- Read Web URLs ---
from typing import Optional
import requests

def fetch_url_content(url: str) -> Optional[str]:
    """
    Fetches content from a URL by performing an HTTP GET request.

    Parameters:
        url (str): The endpoint or URL to fetch content from.

    Returns:
        Optional[str]: The content retrieved from the URL as a string,
                       or None if the request fails.
    """
    prefix_url: str = "https://r.jina.ai/"
    full_url: str = prefix_url + url  # Concatenate the prefix URL with the provided URL
    
    try:
        response = requests.get(full_url)  # Perform a GET request
        if response.status_code == 200:
            return response.content.decode('utf-8')  # Return the content of the response as a string
        else:
            print(f"Error: HTTP GET request failed with status code {response.status_code}")
            return None
    except requests.RequestException as e:
        print(f"Error: Failed to fetch URL {full_url}. Exception: {e}")
        return None
# Replace this with the specific endpoint or URL you want to fetch
url: str = "https://em360tech.com/tech-article/what-is-llama-3"  
content: Optional[str] = fetch_url_content(url)


if content is not None:
    print("Content retrieved successfully:")
else:
    print("Failed to retrieve content from the specified URL.")

# --- Split the texts ---
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from litellm import completion

token_size = 150
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4",
            chunk_size=token_size,
            chunk_overlap=0,
        )

def clean_text(text):
    # Remove all newline characters
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading and trailing spaces
    text = text.strip()
    
    return text

text_chunks = text_splitter.split_text(content)
print(f"Total chunks: {len(text_chunks)}")

def get_embeddings(texts, model="text-embedding-3-small", api_key="your-api-key"):
    # Define the API URL
    url = "https://api.openai.com/v1/embeddings"
    
    # Prepare headers with the API key
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Prepare the request body
    data = {
        "input": texts,
        "model": model
    }
    
    # Send a POST request to the OpenAI API
    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    # Check if the request was successful
    if response.status_code == 200:
        # Return the embeddings from the response
        return response.json()["data"]
    else:
        # Print error if the request fails
        print(f"Error {response.status_code}: {response.text}")
        return None
from langchain_elasticsearch import ElasticsearchStore

def ingest_data_into_es(texts):
    if not es.indices.exists(index=elastic_index_name):
        print("The index does not exist, going to generate embeddings")   
        docsearch = ElasticsearchStore.from_texts( 
            texts,
            embedding = embeddings, 
            es_url = url, 
            es_connection = es,
            index_name = elastic_index_name, 
            es_user = ES_USER,
            es_password = ES_PASSWORD
    )
    else: 
        print("The index already existed")
    
        docsearch = ElasticsearchStore(
            es_connection=es,
            embedding=embeddings,
            es_url = url, 
            index_name = elastic_index_name, 
            es_user = ES_USER,
            es_password = ES_PASSWORD    
        )

    return docsearch
    
docsearch = ingest_data_into_es(text_chunks)

# --- Search for questions ---
def search(str):
    docs = docsearch.similarity_search(str)
    return docs
question = "what is openai o1 model?"
docs = search(question)
print("Found docs: ", len(docs))
print(docs)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

print(format_docs(docs))

# --- Prompts ---
# First prompt will check to see if the retrieved context can answer the user question.
# Second prompt will get the context and question and generates the response.

# First Prompt
decision_system_prompt = """Your job is decide if a given question can be answered with a given context. 
If context can answer the question return 1.
If not return 0.

Do not return anything else except for 0 or 1.

Context: {context}
"""

user_prompt = """
Question: {question}

Answer:"""

# Second Prompt
system_prompt = """You are an expert for answering questions. Answer the question according only to the given context.
If question cannot be answered using the context, simply say I don't know. Do not make stuff up.
Your answer MUST be informative, concise, and action driven. Your response must be in Markdown.

Context: {context}
"""

user_prompt = """
Question: {question}

Answer:"""

# --- Ask questions ---
def azure_openai_completion(question, context, is_system_prompt=False):
    prompt = system_prompt if is_system_prompt else decision_system_prompt
    summary = chat.chat.completions.create(
    model = AZURE_DEPLOYMENT_ID,
    messages=[
            {"role": "system", "content": prompt.format(context=context) },
            {"role": "user", "content": user_prompt.format(question=question)},
        ]
    )

    print(summary)
    return summary
question = "what is openai o1 model"
results = search(question)
context = format_docs(results)
response = azure_openai_completion(question, context)

has_answer = response.choices[0].message.content

question = "what is Llama 3?"
results = search(question)
context = format_docs(results)
response = azure_openai_completion(question, context)

has_answer = response.choices[0].message.content

# --- Check to see if retrieved context can answer the question or not ---
from IPython.display import Markdown, display
from duckduckgo_search import DDGS
def format_search_results(results):
    return "\n\n".join(doc["body"] for doc in results)
    

print(f"Question: {question}")
if has_answer == '1':
    print("Context can answer the question")
    # response = completion(
    #     model="gpt-4o-mini",
    #     messages=[{"content": system_prompt.format(context=context),"role": "system"}, {"content": user_prompt.format(question=question),"role": "user"}],
    #     max_tokens=500
    # )
    response = azure_openai_completion(question, context, True)
    print("Answer:")
    display(Markdown(response.choices[0].message.content))
else:
    print("Context is NOT relevant. Searching online...")
    results = DDGS().text(question, max_results=5)
    context = format_search_results(results)
    print("Found online sources. Generating the response...")
    # response = completion(
    #     model="gpt-4o-mini",
    #     messages=[{"content": system_prompt.format(context=context),"role": "system"}, {"content": user_prompt.format(question=question),"role": "user"}],
    #     max_tokens=500
    # )
    response = azure_openai_completion(question, context, True)
    print("Answer:")
    display(Markdown(response.choices[0].message.content))
    
print(results)

import requests

# URL of the file
url = 'https://chrt.fm/track/46DD7B/media.transistor.fm/7387a8a4/cefc95d5.mp3?download=true&src=player'

# Send a HTTP request to the URL
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Open a local file in binary write mode
    with open('audio_file.mp3', 'wb') as file:
        # Write the content of the response to the file
        file.write(response.content)
    print('File downloaded successfully')
else:
    print('Failed to download file. Status code:', response.status_code)