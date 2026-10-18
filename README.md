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

Requires Python 3.10 or newer (the tests were run on Python 3.13 and 3.14). Check with `python3 --version` (on Windows use `python --version`). The Python that ships with older macOS versions may be too old, so install a newer one if needed.

```bash
git clone https://github.com/harshal1900/helpdesk-copilot-adk.git
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
| `429 RESOURCE_EXHAUSTED` | Quota reached. On the free tier a model may allow only about 20 requests per day per project (`...PerDayPerProject...`, `limit: 20`). Retrying will not help until the quota resets (the message says when). `limit: 0` means that model has no free quota at all. | Wait for the reset, switch the model in `.env` (quota is counted per model), or enable billing on the project. `scripts/measure_routing.py` makes one call per ticket, so its 21 sample tickets alone exceed 20. |
| `404 ... model not found` | The model name in `.env` does not exist for your key. | Copy a current model name from AI Studio into `.env`. |

After editing `.env`, stop `adk web` (Ctrl-C) and start it again.

No key yet? Set `EMBEDDING_BACKEND=local` to try retrieval offline. The three LLM steps still need a key.

## Design notes

- **Fail closed.** If a routing node ever emits a route that no edge matches, ADK only logs a warning and the run ends with no decision. Every routing node therefore has a `DEFAULT_ROUTE` edge that leads to `escalate` (or to `validate`, which escalates an unjudged draft).
- **Per-ticket state is reset in `intake`.** ADK session state persists across messages in one session, as in the `adk web` chat window. Without the reset, an escalated ticket showed the unsent draft of the previous ticket. `tests/test_session_isolation.py` covers this.

## Tests

```bash
pytest
```

The tests run **offline, with no API key**. They run the real workflow graph end to end against a scripted fake model (`tests/fakes.py`), plus unit tests for retrieval, validation rules, eval-file schemas and scoring.

What this proves: the graph wiring, routing, state passing, validation rules and audit trail work as designed.
What it does **not** prove: how well Gemini classifies or drafts. Measure that on your own tickets, as shown below.

## Free-tier quota: plan your runs

On the free tier, each Gemini model allows only a small number of requests per day per project (for example 20 for some Flash models; check yours at https://aistudio.google.com/rate-limit). Quota is counted **per model**. One ticket in `adk web` uses up to 3 calls. The evals use about 12 calls, and routing measurement uses 1 call per ticket. Three ways to stay unblocked:

1. List the models your key can call: `python scripts/check_models.py flash`.
2. Spread the load: set a different model in `.env` for `EVAL_JUDGE_MODEL` and `ROUTING_MODEL` than for the app.
3. Or enable billing on the Google Cloud project behind the key, which raises the limits.

## Measure routing accuracy

```bash
python scripts/measure_routing.py                      # 21 sample tickets in data/
python scripts/measure_routing.py --limit 10           # first 10 only, saves quota
python scripts/measure_routing.py --file my_tickets.csv
```

The model comes from `ROUTING_MODEL` in `.env` (default: `TRIAGE_MODEL`). If a quota or model error stops the run, the script scores the tickets that finished and says how many.

CSV columns: `ticket,queue,ticket_type,priority`. The sample file only shows that the script works. Any accuracy target (for example 95%) must be measured on a large, representative set of your own historical tickets, labelled by your support team.

## ADK evals

`adk eval` needs the eval extra. It is in `requirements.txt`; if you installed earlier, run `pip install "google-adk[eval]"`.

```bash
python scripts/run_evals.py
```

This runs `adk eval` on `evals/helpdesk.evalset.json` with the criteria in `evals/test_config.json` (`final_response_match_v2`, one judge sample per case).

**Changing the judge model.** ADK's default judge, `gemini-2.5-flash`, returns 404 for new API users, so the judge must be named. Set it in `.env`:

```
EVAL_JUDGE_MODEL=gemini-3.8-flash
```

`run_evals.py` writes that name into a copy of the config (`.cache/eval_config.json`) and runs the eval. If you call `adk eval` directly, edit `judge_model` in `evals/test_config.json` instead. Without `EVAL_JUDGE_MODEL`, it uses `JUDGE_MODEL`.

A run makes about 3 model calls per case plus 1 judge call, 12 calls for the three cases. The files are validated against ADK's schemas in the tests. Check the thresholds and extend the evalset with real tickets. You can also capture real sessions into an evalset from the **Eval** tab in `adk web`.

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
scripts/          run_evals.py, measure_routing.py, check_models.py
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

MIT, see `LICENSE`.
