# data/eval/

Evaluation data for `scripts/07_ragas_eval.py`. If you haven't worked with
[RAGAS](https://github.com/explodinggremlins/ragas) or a "golden set" before,
this is a tutorial, not just a schema reference — read it top to bottom once,
then use it as a lookup later.

## Files

- `golden_qa.json` — the golden set: hand-written questions with known-correct
  answers and the exact corpus passages that justify them. Checked into git
  so a run today and a run next month are graded against the same bar.
- `ragas_report.json` — **generated**, not checked in (see `.gitignore`).
  Produced fresh each time you run `07_ragas_eval.py`; it's a snapshot of one
  run, not a fixture to edit by hand.

## Why a "golden set" at all?

Retrieval quality is easy to eyeball on one question and hard to trust in
aggregate. `03_eval.py`'s lexical heuristics (word overlap between answer and
context) catch *some* problems cheaply but can't tell you whether an answer
is actually *correct*, only whether it echoes the retrieved text. A golden
set fixes that by pairing each question with a **known-correct answer**
(`ground_truth`) and the **known-correct supporting passage**
(`reference_contexts`) written by a human ahead of time — so you're grading
the pipeline against ground truth, not against itself.

## Schema

Each entry in `golden_qa.json` is:

```jsonc
{
  "question": "What is unusual about how Kadabra evolves into Alakazam?",
  "ground_truth": "Kadabra evolves into Alakazam via a trade rather than by leveling up or using an evolution stone.",
  "reference_contexts": [
    "**Research Note**: Kadabra's evolution to Alakazam famously requires a trade rather than a level threshold or stone—an anomaly Sabrina's facility studies as evidence that some evolutionary leaps require an external psychological catalyst."
  ]
}
```

- **`question`** — what a user would actually type. Write it the way a
  real person asks, not the way the corpus phrases it — that's the point of
  testing retrieval, not testing whether the wording happens to match.
- **`ground_truth`** — the correct answer, in your own words. This becomes
  RAGAS's `reference` field, used by AnswerRelevancy and ContextRecall (below)
  as the bar the pipeline's actual answer is measured against.
- **`reference_contexts`** — **verbatim** excerpts copied from the corpus
  file, not paraphrased. ContextPrecision and ContextRecall compare the
  pipeline's *actual retrieved chunks* against these strings to judge whether
  retrieval found the right material. A paraphrase breaks that comparison
  silently — the metric will look worse (or, coincidentally, better) than
  retrieval actually is, for reasons that have nothing to do with retrieval
  quality. If you change the wording of a corpus entry, update any
  `reference_contexts` that quote it in the same commit.

## One question, traced end to end

Take the Kadabra→Alakazam example above and follow what `07_ragas_eval.py`
does with it:

1. **Run the real pipeline.** `ChatEngine.process_turn("What is unusual
   about how Kadabra evolves into Alakazam?")` runs the *exact* topic
   gate → decider → retrieve → rerank → compose → LLM flow a chat user would
   hit — this script doesn't shortcut retrieval or reranking to make the eval
   easier. Say it retrieves the Alakazam chunk (correct) plus two unrelated
   chunks (noise), reranks the Alakazam chunk to the top, and the LLM answers
   correctly using it.
2. **Package the outcome.** The actual retrieved chunks' text becomes
   `retrieved_contexts`; the LLM's actual answer becomes `response`. Together
   with the golden set's `ground_truth` (→ `reference`) and
   `reference_contexts`, this becomes one `ragas.SingleTurnSample`.
3. **Score it.** A separate **judge LLM** (configurable via
   `--judge-model`/`--judge-provider` — deliberately independent from the
   pipeline's own LLM, so the model isn't grading its own homework) reads the
   sample and scores four things — explained below.

## What the four metrics actually measure

RAGAS's classic quartet splits into two answer-quality metrics and two
retrieval-quality metrics — worth knowing which is which when a score is low,
since the fix is different (better prompting/model vs. better
chunking/reranking).

**Faithfulness** — *"Did the model make anything up?"* The judge breaks the
generated answer into individual factual claims, then checks each claim
against the retrieved context. The score is the fraction of claims that are
actually supported. A low score means hallucination: the answer states
things the retrieved passages never said, even if the topic is right.

**AnswerRelevancy** — *"Did the model actually answer the question asked?"*
The judge generates several hypothetical questions that the given answer
*would* be a good response to, then measures how similar those are to the
question that was really asked. A high-faithfulness, low-relevancy answer is
the "technically true, but didn't answer what I asked" failure — accurate,
evasive, or off on a tangent.

**ContextPrecision** — *"Of what got retrieved, how much was actually
useful?"* The judge checks each retrieved chunk against the reference answer
and scores whether it was relevant, weighted so relevant chunks ranked near
the top count more than ones buried at the bottom. A low score means the
retriever/reranker is pulling in noise — chunks that don't help, and dilute
or distract the prompt the LLM has to work from. **This is the metric the
reranker upgrade in this PR (cross-encoder over lexical-only) is meant to
improve.**

**ContextRecall** — *"Of what the reference answer needed, how much did
retrieval find?"* The judge checks whether each sentence in `ground_truth`
can be traced back to something in `retrieved_contexts`. A low score means
retrieval *missed* something necessary — even if everything it did retrieve
was on-topic (high precision, low recall is "found some relevant stuff, but
not the piece that actually answers this").

None of the four scores requires a passing bar on their own — read them
together. Good faithfulness with poor context recall, for instance, usually
means the answer correctly says "I don't know" or hedges given a partial
retrieval, which is arguably the *right* behavior even though the number
looks bad in isolation. Look at the per-question breakdown printed by the
script, not just the aggregate, before concluding anything.

## Extending the golden set

Add entries covering content you've added to the corpus, matching the
existing style (one clear question, one factual `ground_truth`, one or more
verbatim `reference_contexts`). A few habits that keep the set useful:

- Prefer questions with one clearly correct answer over open-ended ones —
  RAGAS's judge is measuring against `ground_truth`, and a fuzzy reference
  makes every metric noisier.
- Cover a spread of content types (species stats, trainers, lore, items,
  mechanics) rather than many similar questions about one entry — the
  aggregate score is only as informative as the coverage behind it.
- If a golden question starts failing after a corpus or chunking change,
  that's a real signal worth investigating, not just "update the golden set
  until it passes."

## Running it

```bash
pip install -r requirements.txt -r requirements-eval.txt
python scripts/07_ragas_eval.py
```

Needs an API key for whichever provider you use (`OPENAI_API_KEY`,
`GOOGLE_API_KEY`, or `ANTHROPIC_API_KEY`) — both the pipeline under test and
the judge need one, though by default the judge just reuses the pipeline's.
Costs real money: RAGAS issues several judge calls per metric per question,
so a 15-question run is on the order of a hundred-plus LLM calls. A cheap,
fast model (e.g. `claude-haiku-4-5`) is plenty for the judge role and keeps a
full run cheap — pass it via `--judge-model claude-haiku-4-5 --judge-provider
claude`.
