import streamlit as st
import os
from dotenv import load_dotenv
import tempfile
import pandas as pd

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

load_dotenv()

st.title("📄 Multi-Document Summarizer & Q/A App (RAG + FAISS, LCEL Version)")

# 1️⃣ UPLOAD DOCUMENTS
uploaded_files = st.file_uploader(
    "Upload multiple files (PDF, TXT, CSV, XLSX)",
    type=["pdf", "txt", "csv", "xlsx"],
    accept_multiple_files=True
)

if uploaded_files:
    documents = []

    for file in uploaded_files:
        suffix = file.name.split('.')[-1].lower()

        # --------- PDF & TXT handled via tempfile ----------
        if suffix in ["pdf", "txt"]:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp_file:
                tmp_file.write(file.read())
                tmp_path = tmp_file.name

            if suffix == "pdf":
                loader = PyPDFLoader(tmp_path)
                loaded_docs = loader.load()
            else:
                loader = TextLoader(tmp_path, encoding="utf-8")
                loaded_docs = loader.load()

            # Store original file name in metadata
            for doc in loaded_docs:
                doc.metadata["source"] = file.name
            documents.extend(loaded_docs)

        # --------- CSV ----------
        elif suffix == "csv":
            df = pd.read_csv(file)
            text_data = df.to_string(index=False)
            documents.append(
                Document(page_content=text_data, metadata={"source": file.name})
            )

        # --------- XLSX ----------
        elif suffix == "xlsx":
            xls = pd.ExcelFile(file)
            for sheet in xls.sheet_names:
                df = pd.read_excel(file, sheet_name=sheet)
                text_data = f"Sheet Name: {sheet}\n\n" + df.to_string(index=False)
                documents.append(
                    Document(page_content=text_data, metadata={"source": f"{file.name} - {sheet}"})
                )

    # 2️⃣ SPLIT DOCUMENTS
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    # 3️⃣ EMBEDDINGS + VECTOR DB
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(chunks, embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    # 4️⃣ LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0.2)

    # 5️⃣ PROMPT
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are an intelligent AI assistant. Answer the question using ONLY the provided context.

Context:
{context}

Question:
{question}

If the answer cannot be found in the context, reply:
"Information not available in the provided context."

Answer:
"""
    )

    parser = StrOutputParser()

    # 6️⃣ LCEL RAG CHAIN
    rag_chain = (
        {
            "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | parser
    )

    st.success("Documents processed successfully!")

    # 7️⃣ MODE SELECTION
    mode = st.radio("Choose Mode:", ["Question Answering", "Summarization"])

    # 8️⃣ QUESTION ANSWERING
    if mode == "Question Answering":
        query = st.text_input("Ask your question:")
        if query:
            with st.spinner("Searching in documents..."):
                answer = rag_chain.invoke(query)
            st.write(answer)

    # 9️⃣ SUMMARIZATION
    if mode == "Summarization":
        # Use metadata 'source' for user-friendly labels
        doc_labels = ["All Documents"] + [doc.metadata.get("source", f"Document {i+1}") for i, doc in enumerate(documents)]
        selected_doc = st.selectbox("Select document to summarize:", doc_labels)

        if st.button("Summarize"):
            with st.spinner("Summarizing..."):
                if selected_doc == "All Documents":
                    summary_prompt = "Provide a detailed summary of all uploaded documents."
                    summary = rag_chain.invoke(summary_prompt)
                else:
                    # Summarize only the selected document
                    summary_prompt = f"Provide a detailed summary of ONLY this document: {selected_doc}"
                    summary = rag_chain.invoke(summary_prompt)

            st.write(summary)
