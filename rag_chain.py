from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
import tempfile
import shutil

from keys import INFERENCE_API_KEY
from prompt import TEMPLATE

def load_pdf_text(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        shutil.copyfileobj(uploaded_file, temp_file)
        temp_file_path = temp_file.name

    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    total_text = "\n".join(doc.page_content for doc in docs)
    doc_length = len(total_text)  

    return docs, doc_length

def determine_optimal_chunk_size(doc_length):
    if doc_length < 5000:  
        chunk_size = 500
        chunk_overlap = 100
        return chunk_size, chunk_overlap
    elif doc_length < 20000:  
        chunk_size = 1000
        chunk_overlap = 250
        return chunk_size, chunk_overlap
    else:
        chunk_size = 2000
        chunk_overlap = 500
        return chunk_size, chunk_overlap
    
# def determine_optimal_chunk_size(doc_length, min_chunk=500, max_chunk=2000, min_overlap=100, max_overlap=500): #This function is better but uses more computational power
#     min_length = 0
#     max_length = 20000  

#     if doc_length > max_length:
#         scale_factor = 1
#     else:
#         scale_factor = (doc_length - min_length) / (max_length - min_length)

#     chunk_size = int(min_chunk + (max_chunk - min_chunk) * scale_factor)
#     overlap = int(min_overlap + (max_overlap - min_overlap) * scale_factor)

#     return chunk_size, overlap

def chunk_and_store_in_vector_store(docs, chunk_size, chunk_overlap):
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=INFERENCE_API_KEY, model_name="sentence-transformers/all-MiniLM-l6-v2"
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore

def process_user_input(user_query, vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    llm = HuggingFaceEndpoint(
        huggingfacehub_api_token=INFERENCE_API_KEY,
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        task="text-generation",
        max_new_tokens=512,
        top_k=50,
        top_p=0.8,
        temperature=0.1,
        repetition_penalty=1
    )

    template = TEMPLATE
    custom_rag_prompt = PromptTemplate.from_template(template)

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    llm_response = rag_chain_with_source.invoke(user_query)
    print("\nLLM Response: \n", llm_response)
    final_output = llm_response["answer"].strip()
    print("\nTrimmed LLM Answer: \n", final_output)
    return final_output

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def substring_after(s, delim):
    return s.partition(delim)[2]

