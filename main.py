import re
import numpy as np
import openai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT","")
deployment = "gpt-4o"
model_name = "gpt-4o"

client = openai.AzureOpenAI(
    api_version="2024-12-01-preview",
    api_key=api_key,
    azure_endpoint=endpoint
)


def parse(filepath):
    parsed_string=""
    try:
        with open(filepath,"r",encoding="utf-8") as file:
            for line in file:
                parsed_string += line
    except FileNotFoundError:
        print("File not found.")

    return parsed_string

def chunk(text):
    chunk_list = re.split(r'\n{3,}',text)

    return [c for c in chunk_list if c.strip()]

def embedding(chunk):

    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=chunk
    )
    return [{"chunk": c, "embedding": e.embedding} for c, e in zip(chunk, response.data)]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))  
    
def retrieve(query, embedded_chunks, top_n=3):
    query_embedding = client.embeddings.create(
        model="text-embedding-ada-002",
        input=query
    )
    score_list = []
    for i in embedded_chunks:
        score = cosine_similarity(i["embedding"],query_embedding.data[0].embedding)
        score_list.append({"chunk":i["chunk"],"score":score})
    sorted_list = sorted(score_list, key=lambda x: x["score"], reverse=True)
    return sorted_list[:top_n]

def generate(query, retrieved_chunks):

    retrieved = "\n\n---\n\n".join([i["chunk"] for i in retrieved_chunks])
        

    augmented_prompt = f"You are a helpful assistant. Here is context retrieved from a document : {retrieved} \n Answer the user's question using only this context \nIf the answer isn't in the context, say so"
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": augmented_prompt,
            },
            {
                "role": "user",
                "content": query,
            }
        ],
        max_tokens=4096,
        temperature=1.0,
        top_p=1.0,
        model=deployment
    )


    return response.choices[0].message.content



if __name__ == "__main__":
    filepath = "sample.txt"

    text = parse(filepath)
    chunks = chunk(text)
    embedded = embedding(chunks)
    query = "what is a gaylord?"
    results = retrieve(query, embedded)
    response = generate(query,results)
    print(response)
