import asyncio
import os
import ssl
from typing import Any, Dict, List
import certifi
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import (Colors, log_error, log_header, log_info, log_success, log_warning)

load_dotenv()

ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"]=certifi.where()
os.environ["REQUESTS_CA_BUNDLE"]=certifi.where()

embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        show_progress_bar=True,
        chunk_size=50,
        retry_min_seconds=10
    )

vectorstore = PineconeVectorStore(index_name = "langchain-doc-index", embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_Depth = 5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()

# vectorstore = Chroma(persist_directory="chroma_db", embedding_function = embeddings)


async def index_documents_async(documents: List[Document], batch_size: int = 50):
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"Vectorstore Indexing: preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN,
    )
    batches = [
        documents[i:i+batch_size] for i in range(0, len(documents), batch_size)
    ]
    log_info(
        f"Vectorstore Indexing: created {len(batches)} batches of size {batch_size} for indexing"
    )

    async def add_batch(batch: List[Document], batch_num:int):
        try:
            vectorstore.add_documents(batch)
            log_success(
                f"Vectorstore Indexing: successfully added batch {batch_num}/{len(batches)} with {len(batch)} documents"
            )
        except Exception as e:
            log_error(
                f"Vectorstore Indexing: Failed to add batch {batch_num} - {e}"
            )
            return False
        return True
    
    tasks = [add_batch(batch, i+1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"Vectorstore Indexing: All batches processed successfully ({successful}/{len(batches)})"
        )

    else:
        log_warning(
            f"Vectorstore Indexing: Processed {successful}/{len(batches)} batches successfully"
        )









async def main():
    """Main async function to orchestrate the ingestion process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")
    log_info(
        "TavilyCrawl: Starting to Crawl documentation from https://docs.langchain.com/",
        Colors.PURPLE,  
    )

    res = tavily_crawl.invoke(
        {"url": "https://docs.langchain.com/",
        "max_depth":5,
        "extract_depth":"advanced"}
        # "instructions": "content on langchain, langsmith, langgraph and RAG"}
    )
    all_docs = [
        Document(
            page_content=result["raw_content"],
            metadata={"source": result["url"]}
        )
        for result in res["results"]
        if result.get("raw_content")
    ]
    
    log_success(
        f"TavilyCrawl: successfully crawled {len(all_docs)} URLs from documentation site"
    )
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"Text Splitter: Processing {len(all_docs)} documents with 2000 chunk size and 100 overlap",
        Colors.YELLOW
        )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )

    await index_documents_async(splitted_docs, batch_size=100)

    log_header("PIPELINE COMPLETE")
    log_success("Documentation ingestion pipeline finished successfully")
    log_info("Summary", Colors.BOLD)
    # log_info(f"URLs mapped: {len(site_map['results'])}")
    log_info(f"Document extacted: {len(all_docs)}")
    log_info(f"Chunks created: {len(splitted_docs)}")




if __name__ == "__main__":
    asyncio.run(main())