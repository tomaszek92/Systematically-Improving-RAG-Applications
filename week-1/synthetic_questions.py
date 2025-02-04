import asyncio
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
    "Zmodyfikuj okres w pytaniu (np. zamiast całego roku podaj ostatnie 6 miesięcy)",
    "Dodaj dodatkowy kontekst, np. informację o gatunku filmu",
    "Zmień region/pochodzenie (np. pytanie o filmy wyprodukowane w USA vs. inne kraje)",
]

@retry(stop=stop_after_attempt(1), wait=wait_fixed(10))
async def generate_question(chunk: NetflixChunk, sem: Semaphore) -> ChunkEval:
    async with sem:
        chunk_text = f"{chunk.title}. {chunk.description}"
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": """
                    Wygeneruj przykładowe pytanie, na które można odpowiedzieć na podstawie poniższych informacji o filmie.

                    Informacje:
                    {chunk_text}

                    Zasady:
                    - Nie wykorzystuj bezpośrednio konkretnych wartości, jeśli to możliwe.
                    - Pytanie powinno mieć maksymalnie dwa zdania.
                    - Dostosuj pytanie wg ograniczenia: "{chosen_constraint}".
                    - Pytanie musi być odpowiednie, aby mogło zostać wyszukane za pomocą odpowiednich danych o filmie.
                    """}],
            response_model=Question,
            context={
                "snippet": chunk_text,
                "constraint": random.choice(constraints)},
        )

        return ChunkEval(
            show_id=chunk.show_id,
            question=response.question,
            chunk=f"{chunk.title}. {chunk.description}",
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
    results = asyncio.run(main(concurrency_limit=10, num_samples=2, limit=1))
    print(results)
