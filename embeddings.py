
import logging
from pathlib import Path
import sqlite3
from typing import List

from chromadb import PersistentClient
from chromadb.config import Settings
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema.document import Document
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pandas as pd

logging.basicConfig(
    level = "DEBUG",
    format = '%(asctime)s|p%(process)s|%(pathname)s:%(lineno)d|%(levelname)s|%(message)s'
)

def get_pandas_df(
    database: str = "/Users/rafael.pierre/Downloads/database.sqlite",
    query: str = "SELECT * from content",
    sample_size: int = -1
):
    
    with sqlite3.connect(
        database = database
    ) as con:
        df = pd.read_sql_query(query, con)
        if sample_size > 0:
            df = df.sample(n = sample_size)
        df = df.loc[:, "content"]
        df.to_csv("content.csv")

    return df

def get_summaries(
    text: List[str] = [],
    max_tokens: int = 256,
    temperature: float = 0.3,
    top_p: float = 0.9,
    best_of: int = 10
):
    template = """
        Assume you are a music expert. Please summarize the following album
        review from Pitchfork. Make sure to keep info such as release date,
        musical style, which artists are similar, and the review score.

        Please generate the output as JSON dictionary, like in the example below:

        '''json
            "artist": "michael jackson",
            "review": "this is a review",
            "score": "9.0"
            "similar_to": ["prince", "madonna"]
        '''

        {review}
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["review"]
    )

    llm = OpenAI(
        model = "gpt-3.5-turbo-instruct",
        temperature = temperature,
        top_p = top_p,
        best_of = best_of,
        max_tokens = max_tokens
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    summary = llm_chain.batch([{"review": item} for item in text])

    return summary

def create_or_search_embeddings(
    texts: List[str] = [],
    path: str = "./chroma_db",
    query: str = "",
    n_results: int = 20
) -> str:

    chroma_path = Path(path)
    settings = None
    docs = [Document(page_content = text[:1000]) for text in texts]
    openai_embeddings = OpenAIEmbeddings()

    if chroma_path.exists():
        logging.info(f"Loading persisted embeddings...")
        client = PersistentClient(
            path = path,
            settings = Settings(anonymized_telemetry=False)
        )
        collection = client.get_collection("langchain")
        query_result = collection.query(
            query_embeddings = openai_embeddings.embed_query(query),
            n_results = n_results
        )

        result = query_result["documents"]

    else:
        db = Chroma.from_documents(
            documents = docs,
            embedding = openai_embeddings,
            persist_directory = path,
            client_settings = settings
        )

        result = db.similarity_search(
            query = query,
            k = n_results
        )

        result = [doc.page_content for doc in result]

    return str([
        f"\nReview: {doc}\n"
        for doc in result
    ])

def generate_recommendations(
    query: str = "",
    reviews: str = ""
) -> str:
    
    template = """
    You are an expert in alternative music.

    Based on the Pitchfork reviews below, suggest me a music album that is similar to the
    band or artist "{query}".

    Pick one of the albums mentioned below, and in your recommendation
    please tell me why are you recommending me this album.

    {reviews}

    Include info, like for instance, when this album was released, which musical style is it,
    and also the review score.

    ANSWER:
    """

    prompt = PromptTemplate(
        input_variables=["reviews", "query"],
        template=template
    )
    llm = OpenAI(temperature=0.7, max_tokens = 512)
    llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    
    response = llm_chain.run(
        query = query,
        reviews = reviews
    )

    return response
    

if __name__ == "__main__":

    df = get_pandas_df()
    query = input()
    docs = create_or_search_embeddings(
        texts = list(df.values),
        query = query,
        n_results = 10
    )


