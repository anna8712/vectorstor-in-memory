import os
from dotenv import load_dotenv

from langchain import hub
from langchain_community.document_loaders import TextLoader

from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_openai import ChatOpenAI

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS


from langchain_text_splitters import CharacterTextSplitter

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    load_dotenv()

    pdf_path = "/workspaces/vectorstor-in-memory/react-pdf-for-example.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents=documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_react")

    new_vectorstore = FAISS.load_local(
        "faiss_index_react", embeddings, allow_dangerous_deserialization=True
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(
        OpenAI(),
        retrieval_qa_chat_prompt,
    )
    retrieval_chain = create_retrieval_chain(
        new_vectorstore.as_retriever(),
        combine_docs_chain=combine_docs_chain,
    )

    result = retrieval_chain.invoke(
        input={"input": "Give me the gist of ReAct in 3 sentences"}
    )

    print(result["answer"])
