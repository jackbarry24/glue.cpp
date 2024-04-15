from openai import OpenAI
import numpy as np
from termcolor import colored
import pickle
import json

EMBEDDING_DIM = 1536  # for text-embedding-3-small
client = OpenAI()


def embed(text: str):
    return (
        client.embeddings.create(input=text, model="text-embedding-3-small")
        .data[0]
        .embedding
    )


def cosine_similarity(v1, v2):
    assert len(v1) == len(v2) == EMBEDDING_DIM
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


class VectorDB:
    """
    Poor man's vector database w/ zero optimization (linear search).
    """
    def __init__(self):
        self.corpus = []

    def load_corpus_from_glue(self, input_path: str):
        with open(input_path) as f:
            data = json.load(f)

        for c in data:
            text = c.get("text", "").strip()
            self.corpus.append((text, embed(text)))

    def save_corpus(self, output_path: str = "demo.pkl"):
        with open(output_path, "wb") as f:
            pickle.dump(self.corpus, f)

    def load_corpus(self, input_path: str):
        with open(input_path, "rb") as f:
            self.corpus = pickle.load(f)

    def get_top_k(self, input_embedding, k: int):
        if not self.corpus:
            raise ValueError(
                "Corpus is empty. Please use load_corpus to load & calculate embeddings."
            )

        deltas = [(cosine_similarity(input_embedding, v), t) for t, v in self.corpus]
        return sorted(deltas, key=lambda x: x[0], reverse=True)[:k]


def load_first_time():
    db = VectorDB()
    db.load_corpus_from_glue("glue-output.json")
    db.save_corpus()


def prompt_builder(query, context):
    p = f"USER QUERY: {query}\n\n"
    for i, c in enumerate(context):
        p += f"CONTEXT {i+1}: {c}\n\n"
    return p


system = '''You will be given an input in the form "USER QUERY:...\n\nCONTEXT:...". 
Answer the user query briefly and strictly within the given context'''
def query(query, ctx):
    return (
        client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt_builder(query, ctx)},
            ],
        )
        .choices[0]
        .message.content
    )


if __name__ == "__main__":
    db = VectorDB()
    db.load_corpus("demo.pkl")

    while True:
        x = input("query>>> ")
        if x in ["exit", "quit", "q"]:
            break

        ctx = [c[1] for c in db.get_top_k(embed(x), 3)]
        for i, c in enumerate(ctx):
            print(f"{i+1}.", colored(c, "blue"))
        
        print(colored(query(x, ctx), "green"))
