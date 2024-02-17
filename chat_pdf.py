import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import HuggingFaceHub
from langchain.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
import io

from keys import INFERENCE_API_KEY
from prompt import TEMPLATE

def load_pdf_text(uploaded_file):
    if uploaded_file is not None:
        file_stream = io.BytesIO(uploaded_file.read())
        loader = PyPDFLoader(file_stream)
        docs = loader.load()
        return docs
    else:
        st.error("Please upload a PDF file if the correct format.")

def chunk_and_store_in_vector_store(docs):
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=INFERENCE_API_KEY, model_name="sentence-transformers/all-MiniLM-l6-v2"
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore

def process_user_input(user_query, vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    llm = HuggingFaceHub(
        huggingfacehub_api_token=INFERENCE_API_KEY,
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        task="text-generation",
        model_kwargs={
            "max_new_tokens": 512,
            "top_k": 50,
            "top_p": 0.8,
            "temperature": 0.1,
            "repetition_penalty": 1,
        },
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
    final_output = substring_after(llm_response['answer'], "Helpful Answer:")
    print(final_output.strip())
    st.write("Reply: ", final_output.strip())


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def substring_after(s, delim):
    return s.partition(delim)[2]

def main():
    st.set_page_config("Chat-PDF", "📚🤖")
    st.title("Chat-PDF")

    with st.sidebar:
        st.title("Menu:")
        pdf_doc = st.file_uploader("Upload your PDF here", accept_multiple_files=False)
        if st.button("Process PDF"):
            with st.spinner("Processing... This may take a while based on the size of the PDF"): 

                docs = load_pdf_text(pdf_doc)
                vectorstore = chunk_and_store_in_vector_store(docs)
                st.success("PDF Processed")

    user_query = st.text_input("What question would you like to ask your PDF?")

    if user_query:
        process_user_input(user_query, vectorstore)

if __name__ == "__main__":
    main()

