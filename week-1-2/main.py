import asyncio
import braintrust
import kagglehub
import lancedb
import os
import pandas as pd
from dotenv import load_dotenv
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
from lancedb.table import Table
from pydantic import BaseModel
from rich import print
from typing import Literal, Optional
from itertools import product
import uuid

load_dotenv()

class NetflixChunk(BaseModel):
    show_id: str
    title: str
    type: str = ''  # new field for movie type (e.g., Movie or TV Show)
    director: str = ''
    cast: str = ''
    country: str = ''
    date_added: str = ''
    release_year: int
    rating: str = ''
    duration: str = ''
    listed_in: str = ''
    description: str = ''
    
def create_netflix_chunk(row) -> NetflixChunk:
    def safe_str(value):
        return "" if pd.isna(value) else value

    return NetflixChunk(
        show_id=str(row["show_id"]),
        type=safe_str(row.get("type", "")),
        title=safe_str(row["title"]),
        director=safe_str(row.get("director", "")),
        cast=safe_str(row.get("cast", "")),
        country=safe_str(row.get("country", "")),
        date_added=safe_str(row.get("date_added", "")),
        release_year=int(row["release_year"]),
        rating=safe_str(row.get("rating", "")),
        duration=safe_str(row.get("duration", "")),
        listed_in=safe_str(row.get("listed_in", "")),
        description=safe_str(row.get("description", "")),
    )

def get_or_create_lancedb_table(db: Table, table_name: str, embedding_model: str):
    if table_name in db.table_names():
        print(f"Table {table_name} already exists")
        return db.open_table(table_name)

    func = get_registry().get("openai").create(name=embedding_model)

    class Chunk(LanceModel):
        show_id: str
        chunk: str = func.SourceField()
        vector: Vector(func.ndims()) = func.VectorField()

    table = db.create_table(table_name, schema=Chunk, mode="overwrite")
    path = kagglehub.dataset_download("shivamb/netflix-shows")
    csv_path = os.path.join(path, "netflix_titles.csv")
    df = pd.read_csv(csv_path)
    chunks = [create_netflix_chunk(row) for index, row in df.iterrows()]
    formatted_dataset = [{"show_id": chunk.show_id, "chunk": f"""
        {chunk.title}, {chunk.release_year}, {chunk.country}, {chunk.type}
        {chunk.description}
        """} for chunk in chunks][:100]
    table.add(formatted_dataset)

    # table.create_fts_index("chunk", replace=True)
    print(f"{table.count_rows()} chunks ingested into the database")
    return table


def calculate_mrr(predictions: list[str], gt: list[str]):
    mrr = 0
    for label in gt:
        if label in predictions:
            # Find the relevant item that has the smallest index
            mrr = max(mrr, 1 / (predictions.index(label) + 1))
    return mrr

def calculate_recall(predictions: list[str], gt: list[str]):
    # Calculate the proportion of relevant items that were retrieved
    return len([label for label in gt if label in predictions]) / len(gt)

def retrieve(
    question: str,
    table: Table,
    max_k=25,
    mode: Literal["vector", "fts", "hybrid"] = "vector",
    hooks=None,
):
    results = table.search(question, query_type=mode).limit(max_k)
    return [
        {"show_id": result["show_id"], "chunk": result["chunk"]} for result in results.to_list()
    ]
    
def evaluate_braintrust(input, output, **kwargs):
    predictions = [item["score_id"] for item in output]
    labels = [kwargs["metadata"]["chunk_id"]]

    scores = []
    for metric, score_fn in metrics:
        for subset_k in k:
            scores.append(
                braintrust.Score(
                    name=f"{metric}@{subset_k}",
                    score=score_fn(predictions[:subset_k], labels),
                    metadata={"query": input, "result": output, **kwargs["metadata"]},
                )
            )

    return scores

db = lancedb.connect("./lancedb")
table_small = get_or_create_lancedb_table(
    db, "chunks_text_embedding_3_small", "text-embedding-3-small"
)
table_large = get_or_create_lancedb_table(
    db, "chunks_text_embedding_3_large", "text-embedding-3-large"
)

# Define Our Metrics
metrics = [("recall", calculate_recall), ("mrr", calculate_mrr)]
k = [1, 3, 5, 10, 15, 20, 25, 30, 35, 40]

# Load subset of evaluation queries
evaluation_queries = [
    item for item in braintrust.init_dataset(
        project="Netflix-Shows",
        name="Synthetic-Questions",
    )
]

search_query_modes = ["vector"]

embedding_model_to_table = {
    "text-embedding-3-small": table_small,
    "text-embedding-3-large": table_large,
}

# Run evaluations
evaluation_results = []
experiment_id = str(uuid.uuid4())
for search_mode, embedding_model in product(
    search_query_modes, embedding_model_to_table
):
    # Get model instances
    current_table = embedding_model_to_table[embedding_model]

    # Configure retrieval size
    retrieval_limit = 40

    # Run evaluation
    benchmark_result = braintrust.Eval(
        name="Netflix-Shows",
        experiment_name=f"{experiment_id}-{search_mode}-{embedding_model}",
        task=lambda query: retrieve(
            question=query,
            max_k=retrieval_limit,
            table=current_table,
            mode=search_mode,
        ),
        data=evaluation_queries,
        scores=[evaluate_braintrust],
        metadata={
            "embedding_model": embedding_model,
            "query_mode": search_mode,
            "retrieval_limit": retrieval_limit,
        },
    )

    # Process benchmark results
    performance_scores = benchmark_result.summary.scores
    for metric_name, score_data in performance_scores.items():
        metric_type, top_k = metric_name.split("@")
        evaluation_results.append(
            {
                "metric": metric_type,
                "k": int(top_k),
                "embedding_model": embedding_model,
                "query_type": search_mode,
                "score": score_data.score,
            }
        )