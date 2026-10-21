import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from day1_vectorstore import cosine_similarity, VectorStore # reuse Day-1

load_dotenv()                                     # 1. load the key from .env
_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])   # 2. client, built ONCE at module level

MODEL = "gemini-embedding-001"

# ---------------- Part A: the corpus ----------------
# Theme: college life, and one all-rounder student.
# Built on purpose, not at random. Three kinds of sentence:

CORPUS = [
    # --- 4 paraphrase pairs: same meaning, almost no shared words ---
    "Ravi finished first in his entire engineering batch.",              # 0  pairs with 1
    "He graduated at the very top of the class.",                        # 1

    "She captains the university volleyball team.",                      # 2  pairs with 3
    "Leading the college squad on court comes naturally to her.",        # 3

    "He plays guitar at every campus festival.",                         # 4  pairs with 5
    "Performing on stage during college events is his passion.",         # 5

    "She organises blood donation drives every semester.",               # 6  pairs with 7
    "Arranging charity camps for the community is a regular habit.",     # 7

    # --- 4 keyword traps: heavy word overlap, completely different meaning ---
    "The campus festival was cancelled because of heavy rain.",          # 8   traps 4
    "The volleyball court needs new flooring before the tournament.",    # 9   traps 2
    "The exam hall was freezing and nobody could concentrate.",          # 10  traps 0/1
    "His guitar was stolen from the hostel room last night.",            # 11  traps 4

    # --- 4 filler: unrelated campus noise, gives the ranking something to beat ---
    "The library stays open until midnight during finals week.",         # 12
    "Mess food improved a lot after the new contractor took over.",      # 13
    "Parking near the main gate is full by nine every morning.",         # 14
    "The new lab equipment arrived three weeks behind schedule.",        # 15
]



def embed(texts, task_type="RETRIEVAL_DOCUMENT"):
    """Embed a LIST of strings in ONE batched call.
    Returns a list of vectors (a list of lists), in the same order you sent."""
    resp = _client.models.embed_content(          # 3. the call
        model=MODEL,                              # which model
        contents=texts,                           # 4. pass the whole list — this is the batch
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return [e.values for e in resp.embeddings]     # 5. unpack .embeddings -> each .values

if __name__ == "__main__":
    doc_vecs = embed(CORPUS, task_type="RETRIEVAL_DOCUMENT")#turn sentences into vectors at once and get them back in same order

    store = VectorStore()
    for sentence, vec in zip(CORPUS, doc_vecs):#
        store.add(sentence, vec)

    query = "Which student is academically brilliant?"
    query_vec = embed([query], task_type="RETRIEVAL_QUERY")[0]

    print(f"query: {query}\n")
    for sentence, score in store.search(query_vec, k=3):
        print(f"  {score:.3f}  {sentence}")
