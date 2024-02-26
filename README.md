# QnA-PDF-RAG-LangChain
Chat with your PDF files and ask the most specific questions about them!

![image](https://github.com/yasinda-s/QnA-PDF-RAG-LangChain/assets/60426941/c7173672-8cc0-4afd-bec0-877d6180f09c)

## Features

- Upload a PDF and chat with your PDF!
- Uses free embedding model (all-MiniLM-l6-v2) from hugging face to create embeddings for chunks in the vector store/database.
- As a user asks for a question, the vector store is used as a retreiver, to find the ideal emebeddings based on similarity. 
- The RAG chain uses the sources along with the prompt template to generate a human like response to the user's question using another free model (LLM: Mixtral-8x7B-Instruct-v0.1)

## How to run

```sh
Retrieve a HuggingFace API Key (or any other API Keys based on the models you use: OpenAI, Gemini, etc).
To get the HuggingFace key, visit https://huggingface.co/settings/tokens.

(If you are creating a new venv) :
conda create -n rag python=3.10
conda activate rag

pip install -r requirements.txt
streamlit run streamlit_app.py
```
