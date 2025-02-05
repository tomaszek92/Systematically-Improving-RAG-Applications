import asyncio

if __name__ == '__main__':
    df = load_netflix_data()
    chunks = [create_netflix_chunk(row) for index, row in df.iterrows()]
    questions = asyncio.run(main(concurrency_limit=10, num_samples=2, limit=100))
    for question in questions:
        dataset.insert(
            input=[question.question],
            expected=[question.question],
            metadata={"show_id": question.show_id, "chunk": question.chunk},
        ) 