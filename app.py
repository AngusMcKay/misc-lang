import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
import tempfile

st.set_page_config(page_title="📄 PDF Q&A Bot", page_icon="🤖", layout="wide")

st.title("📄 Chat with Your PDF (FAISS version)")
st.write("Upload a PDF and ask questions about it using OpenAI + FAISS.")

# Sidebar for file upload
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

if uploaded_file:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # 1. Load and split the PDF
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # 2. Create embeddings & FAISS vectorstore
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 3. Build a RetrievalQA chain
    llm = ChatOpenAI(model="gpt-4o-mini")  # or "gpt-3.5-turbo"
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=vectorstore.as_retriever()
    )

    st.success("✅ PDF processed! You can now ask questions.")

    # 4. Chat interface
    # Two columns: left (questions), right (answers)
    col1, col2 = st.columns([2, 3])

    with col1:
        query = st.text_input("Ask a question:")

    with col2:
        st.subheader("Answer will appear here 👇")
        if query:
            with st.spinner("Thinking..."):
                answer = qa_chain.run(query)
            st.markdown(f"**Answer:** {answer}")
