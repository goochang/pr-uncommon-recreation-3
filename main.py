from langchain.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import hub

from dotenv import load_dotenv
import os
load_dotenv()

# 모델 초기화
chat_model = ChatOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model_name="gpt-3.5-turbo-1106", temperature=0.25)

# Load PDF documents using PyPDFLoader with text splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=256)
# PDF 파일 로드. 파일의 경로 입력
loader = PyPDFLoader("docs/ai_trand.pdf")

# 페이지 별 문서 로드
docs = loader.load()
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | chat_model
    | StrOutputParser()
)

response = rag_chain.invoke("라마 3.2 모델이 뭘까!")
print(response)