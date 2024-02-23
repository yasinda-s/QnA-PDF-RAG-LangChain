import streamlit as st
from rag_chain import load_pdf_text, determine_optimal_chunk_size, chunk_and_store_in_vector_store, process_user_input

def create_chat_bubble(text):
    chat_bubble_html = f"""
    <style>
    .chat-bubble {{
        max-width: 100%;
        margin: 10px;
        padding: 10px;
        background-color: #262730;
        border-radius: 16px;
        border: 1px solid #36454F;
    }}
    .chat-container {{
        display: flex;
        flex-direction: column;
        align-items: flex-start;
    }}
    </style>
    <div class="chat-container">
        <div class="chat-bubble">
            {text}
        </div>
    </div>
    """
    return chat_bubble_html

def main():
    st.set_page_config("ConvoPDF")
    st.title("ConvoPDF 📚🤖")

    if 'vectorstore' not in st.session_state:
        st.session_state['vectorstore'] = None

    with st.sidebar:
        st.title("Get started with ConvoPDF:")
        pdf_doc = st.file_uploader("Upload your PDF", accept_multiple_files=False)
        if st.button("Process PDF"):
            with st.spinner("Processing... This may take a while based on the size of the PDF"):
                if pdf_doc is not None:
                    docs, doc_length = load_pdf_text(pdf_doc)
                    chunk_size, chunk_overlap = determine_optimal_chunk_size(doc_length)
                    st.session_state['vectorstore'] = chunk_and_store_in_vector_store(docs, chunk_size, chunk_overlap)
                    st.success("PDF Processed")
                else:
                    st.error("No PDF file uploaded. Please upload a PDF file.")

    user_query = st.text_input("What question would you like to ask your PDF?")

    if user_query and st.session_state['vectorstore']:
        llm_answer = process_user_input(user_query, st.session_state['vectorstore'])
        st.markdown(create_chat_bubble(llm_answer), unsafe_allow_html=True)

    elif user_query:
        st.warning("Please upload a PDF and process it before asking a question")

if __name__ == "__main__":
    main()

