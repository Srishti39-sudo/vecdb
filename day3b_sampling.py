"""Day 3b: temperature and top-p on the generation step of your RAG.

Retrieval stays exactly as it is (day3_rag.retrieve). Only the LLM call changes.

  temperature  how sharp the next-word choice is.   0 = always the top word (same answer every time)
                                                     1 = normal, 2 = wild
  top_p        which words are allowed at all.      0.1 = only the few most likely words
                                                     1.0 = every word is allowed

You fill in 2 TODOs. Everything else runs.

  .venv/bin/python day3b_sampling.py "how does learning change the brain?"

Quota: each run makes 9 generate_content calls (3 settings x 3 repeats). Free tier = 20 a day.
"""
import sys
from google.genai import types                      # GenerateContentConfig lives here
from google.genai import errors as genai_errors

from day3_rag import retrieve, PROMPT, CHAT_MODEL, NO_ANSWER
from day2_embeddings import _client


def chat(prompt, temperature, top_p):
    """One Gemini call with explicit sampling settings."""
    try:
        # TODO 1: pass temperature and top_p to Gemini.
        #   Build a config and hand it to generate_content:
        #       cfg = types.GenerateContentConfig(temperature=..., top_p=...)
        #       resp = _client.models.generate_content(model=CHAT_MODEL, contents=prompt, config=...)
        resp = _client.models.generate_content(model=CHAT_MODEL, contents=prompt)
    except genai_errors.ClientError as e:
        return f"[api error {e.code}] {e.message}"
    return resp.text.strip()


def ask(question, temperature, top_p, k=5):
    hits = retrieve(question, k=k)
    if not hits:
        return NO_ANSWER
    context = "\n".join(f"- {fact}" for fact, _ in hits)
    return chat(PROMPT.format(context=context, question=question), temperature, top_p)


SETTINGS = [            # (name, temperature, top_p)
    ("focused", 0.0, 1.0),
    ("normal",  1.0, 0.95),
    ("wild",    1.8, 1.0),
]
REPEATS = 3


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "how does learning change the brain?"
    print(f"query: {q}\n")
    for name, t, p in SETTINGS:
        answers = []
        # TODO 2: ask the same question REPEATS times with this setting and collect the answers.
        #   for _ in range(REPEATS):
        #       answers.append(ask(q, t, p))
        print(f"--- {name}: temperature={t} top_p={p}  ->  {len(set(answers))} different answers out of {len(answers)}")
        for a in answers:
            print("   ", a)
    print("\nWhat you should see: 'focused' gives the same answer every time; 'wild' drifts.")
    print("For a RAG assistant that must stick to the facts, which setting would you ship?")
