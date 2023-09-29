
import logging
from pathlib import Path
import sqlite3
from typing import List

from chromadb import PersistentClient
from chromadb.config import Settings
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pandas as pd

logging.basicConfig(
    level = "DEBUG",
    format = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
)

def get_pandas_df(
    database: str = "/Users/rafael.pierre/Downloads/database.sqlite",
    query: str = "SELECT * from content",
    sample_size: int = 10
):
    
    with sqlite3.connect(
        database = database
    ) as con:
        df = pd.read_sql_query(query, con)
        df = df.sample(n = sample_size)
        df = df.loc[:, "content"]
        df.to_csv("content.csv")

    return df

def save_embeddings(
        texts: List[str],
        path: str = "./chroma_db"
    ) -> Chroma:

    chroma_path = Path(path)
    settings = None
    docs = [Document(page_content = text[:1000]) for text in texts]
    openai_embeddings = OpenAIEmbeddings()

    if chroma_path.exists():
        client = PersistentClient(
            path = "./chroma_db",
            settings = Settings(anonymized_telemetry=False)
        )
        settings = client.get_settings()

    db = Chroma.from_documents(
        documents = docs,
        embedding = openai_embeddings,
        persist_directory="./chroma_db",
        client_settings = settings
    )

    return db

if __name__ == "__main__":

    df = get_pandas_df()
    db = save_embeddings(texts = list(df.values))
    query = "Which bands are similar to Slayer?"
    reviews = str([
        f"\nReview: {doc.page_content}\n"
        for doc in db.similarity_search(
            query = query,
            k = 5
        )
    ])
    
    template = """
    You are an expert in music.

    Based on the Pitchfork reviews below, suggest me a music album.
    Pick one of the albums mentioned below, and in your recommendation
    please tell me why are you recommending me this album.

    {reviews}

    ANSWER:
    """

    prompt = PromptTemplate(input_variables=["reviews"], template=template)
    llm = OpenAI()
    llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    response = llm_chain.run(reviews)
    print(response)