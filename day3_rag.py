"""
Day 3 — RAG (Retrieval-Augmented Generation)
Full loop: retrieve relevant facts → feed to Gemini → grounded answer.
"""
import logging

from google.genai import errors as genai_errors

from day1_vectorstore import VectorStore
from day2_embeddings import embed          # reuse your Day-2 Gemini embedder

# the SDK warns about automatic function calling on every generate_content call.
# we do not use function calling, so the warning is just noise.
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

# --- knowledge base: 16 facts on ONE topic (neuroscience) ---


CORPUS = [
    # one fact per line about how and why the brain works; each line stands alone
    "The human brain contains about 86 billion neurons and roughly as many glial cells that support them.",
    "The brain is about 2 percent of body weight but uses roughly 20 percent of the body's energy at rest.",
    "Neurons talk to each other at synapses by releasing chemical messengers called neurotransmitters, such as glutamate, GABA and dopamine.",
    "A neuron fires an all-or-nothing electrical spike called an action potential; stronger signals are coded by firing more often, not by bigger spikes.",
    "Myelin, a fatty sheath made by glial cells, insulates nerve fibres so signals can travel at over 100 metres per second.",
    "Learning physically rewires the brain: synapses that are used together grow stronger, often summed up as 'neurons that fire together wire together'.",
    "The hippocampus is needed to form new long-term memories of events; the patient H.M. lost this ability after it was removed but could still learn new motor skills.",
    "During sleep the brain replays and consolidates the day's memories, and studies in mice suggest it also clears metabolic waste more efficiently while asleep.",
    "Dopamine signals reward prediction error, the gap between the reward we expected and the one we got, which drives learning and motivation more than pleasure itself.",
    "The prefrontal cortex supports planning, self-control and working memory, and it is one of the last brain regions to mature, continuing into the mid-twenties.",
    "The amygdala helps detect threats and tags emotional experiences, which is why emotional events are remembered more vividly.",
    "A leading theory holds that the brain is a prediction machine: perception blends incoming senses with the brain's expectations of what it should see and hear.",
    "Neuroplasticity lasts a lifetime: adults can learn new languages and skills, and rehabilitation can help the brain regain function after a stroke.",
    "Working memory can hold only about four chunks of information at once, which is why grouping information into chunks helps us remember it.",
    "Attention is a limited resource: switching between tasks costs time and accuracy compared with doing one thing at a time.",
    "The default mode network is a set of brain regions most active during rest, daydreaming and thinking about ourselves and other people.",
]


# --- build the store (Task 1) ---
store = VectorStore()
doc_vecs = embed(CORPUS, task_type="RETRIEVAL_DOCUMENT")   # embed all at once
for fact, vec in zip(CORPUS, doc_vecs):
    store.add(fact, vec)

print(f"stored {len(store.items)} facts")

# --- Task 2: retrieve ---
# The similarity floor. Cosine on these embeddings never reaches 0: unrelated
# text still scores ~0.58 because every string lands in a narrow cone of the
# space. Anything at or below that is noise, so we refuse it instead of
# returning the least-irrelevant fact. Measured with calibrate_floor() below --
# do NOT copy this number to another corpus or another embedding model.


# Junk queries used to measure the noise level. None of them is answerable from
# CORPUS, so whatever they score IS the floor.
NOISE_PROBES = [
    "clade",
    "zxqw plonk vermilion 88",
    "the mitochondria is the powerhouse of the cell",
    "https://example.com/photo-of-a-cat-9281.jpg",
    "how do I renew a passport in Ontario",
    "quarterly revenue guidance for the semiconductor sector",
    "recipe for sourdough starter",
    "def __init__(self, *args, **kwargs):",
    "?????",
]


def calibrate_floor(probes=NOISE_PROBES, margin=0.05):
    """Measure the noise level of THIS corpus and suggest a floor.

    One batched embedding call. Run it once per corpus, read the number,
    hard-code it in FLOOR. Do not run it per query.
    """
    vecs = embed(probes, task_type="RETRIEVAL_QUERY")
    tops = [store.search(v, k=1)[0][1] for v in vecs]
    worst = max(tops)
    return {
        "noise_min": min(tops),
        "noise_max": worst,          # the highest score pure junk achieved
        "suggested_floor": round(worst + margin, 2),
    }


# Auto-calibrate on import. Convenient while the corpus keeps changing, but it
# costs one extra embedding call every time this module is imported or run.
# Once the corpus settles, replace this with the measured constant:
#     FLOOR = 0.67
FLOOR = calibrate_floor()["suggested_floor"]


def retrieve(query, k=5, floor=None):
    """Embed the question, return up to k facts scoring above `floor`.

    Returns [] when nothing clears the floor -- that is the point. Pass
    floor=0 to see the raw ranking.
    """
    floor = FLOOR if floor is None else floor
    query_vec = embed([query], task_type="RETRIEVAL_QUERY")[0]
    hits = store.search(query_vec, k=k)
    return [(fact, score) for fact, score in hits if score >= floor]


# --- Task 3: generate a grounded answer ---
from day2_embeddings import _client                # reuse the same Gemini client

CHAT_MODEL = "gemini-3.5-flash"   # gemini-3.5-flash-lite is cheaper and also works

PROMPT = """Answer the question using ONLY the context below.

Rules:
- Answer directly in one or two short sentences. Lead with the single most direct fact.
- Be concrete and specific. Name the thing (job title, number, place, language).
- Do not pad the answer with atmosphere, adjectives, or facts the question did not ask for.
- If the context does not contain the answer, say "I don't know from the given facts."

Context:
{context}

Question: {question}
Answer:"""


def _chat(prompt):
    """One call to the chat model. Returns the text, or a one-line note if the
    API refuses, so a quota error does not bury the run in a traceback."""
    try:
        resp = _client.models.generate_content(model=CHAT_MODEL, contents=prompt)
    except genai_errors.ClientError as e:
        if e.code == 429:
            return (f"[quota] the free-tier daily limit for {CHAT_MODEL} is used up. "
                    "retrieval still works; try again tomorrow or use a billed key.")
        return f"[api error {e.code}] {e.message}"
    return resp.text


NO_ANSWER = "I don't know from the given facts."


def ask(question, k=5):
    hits = retrieve(question, k=k)
    if not hits:
        # Nothing cleared the floor. Answering would mean handing the model
        # five irrelevant facts and hoping it refuses. Skip the API call.
        return NO_ANSWER, []
    context = "\n".join(f"- {fact}" for fact, _ in hits)
    return _chat(PROMPT.format(context=context, question=question)), hits


# --- Task 6: the baseline — same model, NO facts ---
def generate(question):
    """Ask Gemini the bare question, with no retrieved context."""
    return _chat(question)


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]

    # --baseline turns the no-facts comparison back on. It costs a second
    # generate_content call per question, and the free tier allows only 20 a day.
    show_baseline = "--baseline" in args

    # --calibrate measures the noise level of this corpus and exits.
    if "--calibrate" in args:
        print("\ncalibrating floor against", len(NOISE_PROBES), "junk queries...")
        for key, val in calibrate_floor().items():
            print(f"  {key:<16} {val}")
        print(f"  current FLOOR     {FLOOR}")
        raise SystemExit

    args = [a for a in args if a not in ("--baseline", "--calibrate")]

    if args:                                   # python day3_rag.py "your question"
        questions = [" ".join(args)]
    else:                                      # or just run it and type
        questions = iter(lambda: input("\nquery> ").strip(), "")

    for q in questions:
        if not q:
            continue

        print(f"\n{'=' * 60}")
        print(f"query: {q}")

        # --- baseline: plain Gemini, no facts ---
        if show_baseline:
            print("\nWITHOUT facts (plain Gemini):")
            print(generate(q))

        # --- your RAG system: retrieve, then answer from those facts only ---
        answer, hits = ask(q)
        print("\nretrieved:")
        if hits:
            for fact, score in hits:
                print(f"  {score:.3f}  {fact}")
        else:
            print(f"  (nothing scored above the floor of {FLOOR} -- no API call made)")
        print(f"\nWITH facts (RAG):\n{answer}")
