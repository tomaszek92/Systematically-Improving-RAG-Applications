import asyncio
import braintrust
import instructor
import kagglehub
import logging
import os
import openai
import pandas as pd
import random
from dotenv import load_dotenv
from pydantic import BaseModel
from rich import print
from asyncio import Semaphore
from tenacity import retry, stop_after_attempt, wait_fixed

logging.basicConfig(level=logging.INFO)

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
client = instructor.from_openai(openai.OpenAI())

dataset = braintrust.init_dataset(
    project="Netflix-Shows",
    name="Synthetic-Questions",
)

def load_netflix_data() -> pd.DataFrame:
    path = kagglehub.dataset_download("shivamb/netflix-shows")
    print("Path to dataset files:", path)
    csv_path = os.path.join(path, "netflix_titles.csv")
    df = pd.read_csv(csv_path)
    print("Przykładowe dane Netflixa:")
    print(df.head())
    return df

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

class Question(BaseModel):
    chain_of_thought: str = None
    question: str

class ChunkEval(BaseModel):
    show_id: str
    question: str
    chunk: str  # przykładowo możemy przekazać tytuł + opis filmu

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

constraints = [
    "If there's a time period mentioned in the snippet, modify it slightly (e.g., if the snippet refers to the entire year, change it to 6 months or 1.5 years).",
    "Add in some irrelevant context (e.g., mention the weather, a random event, or a backstory that isn't in the snippet).",
    "Change the value of the filter (e.g., if the snippet focuses on results in Canada, change the question to ask about another country or city).",
    "Rephrase the question to focus on a different aspect of the movie (e.g., instead of asking about the plot, ask about the director’s influence).",
    "Compare the movie with another similar movie instead of focusing on it alone.",
    "If a country is mentioned, replace it with a neighboring or culturally similar country.",
    "Frame the question from a different perspective (e.g., how a critic vs. an audience member might interpret it).",
    "Introduce an unexpected but plausible detail, such as a fan theory or speculation.",
    "Instead of asking about the movie directly, ask about the broader genre or trend it belongs to.",
    "Pose a hypothetical scenario (e.g., 'How would the story change if set in a different decade?').",
    "If the snippet contains a numerical fact, slightly alter it to test robustness.",
    "Make the question more open-ended rather than fact-based (e.g., instead of 'What year was it released?', ask 'How did its release impact the industry?').",
]

@retry(stop=stop_after_attempt(2), wait=wait_fixed(10))
async def generate_question(chunk: NetflixChunk, sem: Semaphore) -> ChunkEval:
    async with sem:
        chunk_text = f"""
        {chunk.title}, {chunk.release_year}, {chunk.country}, {chunk.type}
        {chunk.description}
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": """
                    Generate a sample question that can be answered based on the following movie information.

                    Information:
                    {snippet}

                    Rules:
                    - Avoid directly using specific values if possible.
                    - The question should have at most two sentences.
                    - Adjust the question according to the following constraint: "{chosen_constraint}".
                    - The question must be suitable for querying the movie data effectively.
                    """
                }
            ],
            response_model=Question,
            context={
                "snippet": chunk_text,
                "chosen_constraint": random.choice(constraints)
            },
        )

        return ChunkEval(
            show_id=chunk.show_id,
            question=response.question,
            chunk=chunk_text,
        )

async def main(
        concurrency_limit: int = 10,
        num_samples: int = 2,
        limit: int = 50) -> list[ChunkEval]:
    sem = Semaphore(concurrency_limit)
    coroutines = []
    selected_chunks = chunks[:limit]

    for chunk in selected_chunks:
        for _ in range(num_samples):
            coroutines.append(generate_question(chunk, sem))

    results: list[ChunkEval] = await asyncio.gather(*coroutines)
    return results

if __name__ == '__main__':
    df = load_netflix_data()
    chunks = [create_netflix_chunk(row) for index, row in df.iterrows()]
    questions = asyncio.run(main(concurrency_limit=10, num_samples=2, limit=100))
    for question in questions:
        dataset.insert(
            input=question.question,
            expected=[question.question],
            metadata={"show_id": question.show_id, "chunk": question.chunk},
        )