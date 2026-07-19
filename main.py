import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

load_dotenv()
print("Initializing components")

embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )

llm = ChatOpenAI(
    model="openai/gpt-oss-120b", # Always specify an OpenRouter model identifier
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)


vectorstore = PineconeVectorStore(
    index_name = os.environ.get("INDEX_NAME"), embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
    
    {context}

    Question: {question}

    Provide a detailed answer:
    """
)

def format_docs(docs):
    """Format retrived docs into single string"""
    return "\n\n".join(doc.page_content for doc in docs)

def retrieval_chain_without_lcel(query:str):
    docs = retriever.invoke(query)
    context = format_docs(docs)
    messages = prompt_template.format_messages(context = context, question=query)

    response = llm.invoke(messages)

    return response.content

def retrieval_chain_with_lcel():
    retrieval_chain = (
        RunnablePassthrough.assign(context = itemgetter("question")|retriever|format_docs)
        |prompt_template|llm|StrOutputParser()
    )
    return retrieval_chain


if __name__ == "__main__":
    print("Retrieving....")

    query = "What is pinecone in machine learning?"

# =============================================
# Option 1 -  No RAG
# =============================================

print("\n"+"="*100)
print("IMPLEMENTATION 1 - No RAG")
print("="*100)
result_raw = llm.invoke([HumanMessage(content=query)])
print("\nAnswer:\n")
print(result_raw.content)
# =============================================
# Option 2 -  RAG without LCEL
# =============================================
print("\n"+"="*100)
print("IMPLEMENTATION 2 - RAG without LCEL")
print("="*100)
result_without_lcel = retrieval_chain_without_lcel(query)
print("\nAnswer:\n")
print(result_without_lcel)
# =============================================
# Option 3 -  RAG with LCEL
# =============================================
print("\n"+"="*100)
print("IMPLEMENTATION 3 - RAG with LCEL")
print("="*100)
chain_with_lcel = retrieval_chain_with_lcel()
result_with_lcel = chain_with_lcel.invoke({"question": query})
print("\nAnswer:\n")
print(result_with_lcel)
