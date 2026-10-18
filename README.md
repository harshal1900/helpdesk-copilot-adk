# Helpdesk Copilot (Google ADK 2.x)

A small, readable multi-agent workflow that triages a support ticket, drafts a first reply **only from your knowledge base**, checks every claim, and escalates to a human when it is not sure. Built with [Google Agent Development Kit (ADK) 2.x](https://adk.dev/) and Gemini.

This repo accompanies a LinkedIn article that explains the code step by step. Everything here runs on your own API key. No secrets are included.

## What it does

```
ticket
  -> intake -> retrieve (embeddings + cosine search over kb/)
  -> triage agent (LLM)      queue, type, priority, confidence
  -> route_triage (Python)   security / P1 / low confidence  -> escalate
  -> draft agent (LLM)       reply + every claim with source IDs
  -> judge agent (LLM)       is each claim supported by the passages?
  -> validate (Python)       citations, judge verdicts, banned phrases, invented numbers
        all checks pass -> deliver      any check fails -> escalate to a human
```

Design rule: **LLMs only where language understanding is needed** (triage, draft, judge). Routing, validation and escalation are ordinary Python, so they behave the same every time and are easy to test.

Every step appends to an **audit trail** (`state["trail"]`), returned in the final output and optionally written to `AUDIT_DIR` as JSON.

## Quick start

Requires Python 3.10 or newer. Check with `python3 --version` (on Windows use `python --version`). The Python that ships with older macOS versions may be too old, so install a newer one if needed.

```bash
git clone <your-repo-url> helpdesk-copilot-adk
cd helpdesk-copilot-adk
python3 -m venv .venv && source .venv/bin/activate   # Windows: python -m venv .venv, then .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # then edit .env and add your own GOOGLE_API_KEY
adk web .                   # open the URL it prints, pick "helpdesk_copilot"
```

Get a key at <https://aistudio.google.com/apikey>. `.env` is git-ignored, so you can commit and push safely. Only `.env.example` is tracked.

Try these tickets in the web UI:

- `I bought the annual plan 10 days ago and want a refund. Is that possible?` (should auto-reply)
- `Someone logged in to my account from another country.` (should escalate: security, P1)
- `it is not working, please fix` (should escalate: vague)

`adk web` is for development only, not for production.

### Models and free-tier limits

Model names live in `.env` (`TRIAGE_MODEL`, `DRAFT_MODEL`, `JUDGE_MODEL`, `EMBEDDING_MODEL`). Google changes model IDs and free-tier access often. If you get a 404 or a `429 ... limit: 0` error, check the current model list in AI Studio and edit `.env`. No code change is needed.

**Troubleshooting Gemini errors**

| Error in the UI | Meaning | What to do |
| --- | --- | --- |
| `503 UNAVAILABLE ... high demand` | Google's model is busy. Usually temporary. The app already retries `LLM_RETRY_ATTEMPTS` times. | Wait a few minutes and try again, or switch the model names in `.env`. |
| `429 RESOURCE_EXHAUSTED` | Rate limit reached. If the message says `limit: 0`, that model has no free-tier quota for your project. | Wait, or pick a model that has a free tier, or enable billing. |
| `404 ... model not found` | The model name in `.env` does not exist for your key. | Copy a current model name from AI Studio into `.env`. |

After editing `.env`, stop `adk web` (Ctrl-C) and start it again.

No key yet? Set `EMBEDDING_BACKEND=local` to try retrieval offline. The three LLM steps still need a key.

## Tests

```bash
pytest
```

The tests run **offline, with no API key**. They run the real workflow graph end to end against a scripted fake model (`tests/fakes.py`), plus unit tests for retrieval, validation rules, eval-file schemas and scoring.

What this proves: the graph wiring, routing, state passing, validation rules and audit trail work as designed.
What it does **not** prove: how well Gemini classifies or drafts. Measure that on your own tickets, as shown below.

## Measure routing accuracy

```bash
python scripts/measure_routing.py                      # 21 sample tickets in data/
python scripts/measure_routing.py --file my_tickets.csv
```

CSV columns: `ticket,queue,ticket_type,priority`. The sample file only shows that the script works. Any accuracy target (for example 95%) must be measured on a large, representative set of your own historical tickets, labelled by your support team.

## ADK evals

```bash
adk eval helpdesk_copilot evals/helpdesk.evalset.json --config_file_path=evals/test_config.json --print_detailed_results
```

`evals/` holds a starter evalset (three cases) and a config using the LLM-judged `final_response_match_v2` criterion. The files are validated against ADK's schemas in the tests, but the live judge run needs your key and was **not** executed when this repo was prepared. Run it, check the thresholds, and extend the evalset with real tickets. You can also capture real sessions into an evalset from the **Eval** tab in `adk web`.

## Project layout

```
helpdesk_copilot/
  agent.py        the Workflow graph (edges, agents)
  nodes.py        deterministic nodes: intake, retrieve, route_triage, validate, deliver, escalate
  validation.py   the checks that decide deliver vs escalate
  retrieval.py    chunking, embeddings, cosine search
  schemas.py      Pydantic models for every LLM output
  prompts.py      agent instructions
  config.py       environment-based settings
  kb/             sample knowledge base (fictional company, replace with your own)
tests/            offline tests with a scripted fake model
evals/            starter ADK evalset + config
scripts/          routing-accuracy measurement
data/             sample labelled tickets
.env.example      copy to .env and add your own key
```

## Use your own knowledge base

Put markdown files in `helpdesk_copilot/kb/` (or set `KB_DIR`). Start each file with a `# Title` and split it into `## Section` blocks. Each section becomes one passage with an ID such as `refund-policy#1`. Delete `.cache/` if you change embedding models.

## Honest limits

- The validator **reduces and detects** unsupported claims. It cannot guarantee zero hallucination. The judge is also a model, and the number check is simple.
- Similarity thresholds (`MIN_RETRIEVAL_SCORE`) depend on the embedding model. Tune them on your data.
- The number check compares plain numbers and amounts, and does not understand units or dates.
- Tickets and audit files can contain personal data. Protect or purge `AUDIT_DIR`.
- This is a teaching project. Before production, add authentication, rate limiting, monitoring, a human feedback loop and a security review.

## License

MIT (see `LICENSE`). Replace the copyright holder name with your own.
