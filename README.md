# Fieldhand: intake agent + policy reviewer (Milestone II)

Fieldhand is an agentic back office for solo pest control operators. A customer texts the business; the
**intake agent** checks who is texting, confirms the service area, prices the job from the price book,
finds open times (checking the forecast for outdoor work), and proposes a booking or hands the job to the
owner. A second agent, the **policy reviewer**, checks every reply before it is sent, and the owner must
approve every booking before it touches the calendar.

Built with LangGraph and the Gemini API. The business ("Oakline Pest Co.") and all customer data are fictional.

## What changed since Milestone I

| Area | Milestone I | Milestone II |
|---|---|---|
| Agents | One intake agent | Intake agent + policy reviewer (no tools; checks every draft) |
| Human gate | None | `book_appointment` pauses for owner approval; the app shows the exact arguments with Approve / Reject |
| Access | Local only | Public Streamlit link; every visitor gets a private sandbox copy of the database |
| Caps | Model calls per message | Also tool calls per message, output tokens per answer, messages per session, message length, a daily cap |
| Evaluation | Three hand-run scenarios | 14 cases with known-good answers, each run 3 times, pass@1 and pass^3 |

## Quick start

Needs Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then paste your key after GOOGLE_API_KEY=
python -m pytest -q                # 18 offline tests, no key needed
streamlit run app.py               # the app
python run_demo.py                 # one run in the terminal (approves at the gate; --reject rejects)
```

## Evaluation harness

```bash
python -m evals.run_eval                       # all 14 cases x 3 trials, live Gemini
python -m evals.run_eval --cases C01,C11       # a subset
python -m evals.run_eval --resume evals/results/<folder>   # continue after a rate limit
python -m evals.run_eval --dry-run --cases C01,C14         # plumbing check, no key
```

Cases live in `evals/cases.json` (each has a `why`). Every trial gets a fresh copy of the seeded
database, a fixed date (Wednesday, October 21, 2026) and a fixed forecast (Thursday 70% rain), so each
case has one known-good answer. Bookings are checked in the database, not in the agent's words.
Results go to `evals/results/<timestamp>/`: `summary.md` (the table), `results.json`, and one trace per trial.

For a case with n trials and c passes: pass@1 = c / n, and pass^k = C(c, k) / C(n, k), the chance that
k tries in a row all succeed. With n = k = 3, pass^3 is 1 only when all three trials pass.

## How it works

```
START -> agent --(tool calls)----------------> tools -> agent
               --(book_appointment proposed)--> GATE --(owner approves)--> tools -> agent
                                                     --(owner rejects)---> agent
               --(draft reply)----------------> reviewer --(pass)----> finish -> END
                                                         --(revise)--> agent      (once)
                                                         --(blocked)-> guard  -> END
               --(any cap reached)-------------> guard -> END
```

| Tool | What it does | Rule enforced in code |
|---|---|---|
| `lookup_customer` | Record and job history for the sender | Uses the sender's number from graph state; the model cannot ask for anyone else |
| `check_service_area` | Is the ZIP served? | 14 intown Atlanta and Decatur ZIPs |
| `get_quote` | Price from the price book | Size tiers; commercial jobs and bed bugs go to the owner |
| `find_open_slots` | Open 2-hour windows | Skips booked windows; outdoor work checks api.weather.gov for 50%+ rain |
| `book_appointment` | Writes the job to the calendar (**gated**) | Owner approval first; price book price, no double booking, no rain days, no same day |
| `request_owner_approval` | Files a ticket in the owner's inbox | Discounts, complaints, commercial, anything off the price book |

**Why a second agent.** Separation of duties. The intake agent reads untrusted customer text and holds
the tools, so a prompt injection can steer what it writes. The reviewer never sees the customer's
message as instructions and has no tools; it judges only the draft against the policy and the tool
results. Code checks it cannot be talked out of: every `$` amount and `FH-` confirmation number must
appear in a tool result, no other customer's name, street, or gate code, no safety promises, and a
length limit. A draft that fails goes back once; a second failure ends in the guard's hand-off reply.

**Caps** (environment variables, defaults in `fieldhand/config.py`): model calls per message
`FIELDHAND_MAX_ITERATIONS=8`, tool calls per message `FIELDHAND_MAX_TOOL_CALLS=10`, output tokens per
answer `FIELDHAND_MAX_OUTPUT_TOKENS=2048`, messages per session `FIELDHAND_MAX_TURNS=12`, characters per
message `FIELDHAND_MAX_CHARS=600`, messages per day for the whole app `FIELDHAND_DAILY_CAP=400`.

## Deploy (Streamlit Community Cloud)

1. Push this folder to a GitHub repository. Do not commit `.env`.
2. At https://share.streamlit.io choose **Create app**, pick the repo, main file `app.py`.
3. Under **Advanced settings**, choose Python 3.12 or 3.13 and paste the secrets:
   ```toml
   GOOGLE_API_KEY = "your key"
   FIELDHAND_MODEL = "the model ID from your .env"
   FIELDHAND_CONTACT = "your email"
   ```
4. Deploy, then open the link in a private window to check it works without a login.

The key is read from Streamlit secrets into environment variables. It never appears in the UI or the repo.

## Files

```
fieldhand/agent.py      LangGraph graph: agent, human gate, reviewer, guard, start_turn/resume_turn
fieldhand/critic.py     the policy reviewer (second agent)
fieldhand/tools.py      the six tools and the booking price preview shown at the gate
fieldhand/prompts.py    system prompt
fieldhand/config.py     settings and caps
fieldhand/db.py         SQLite schema and synthetic seed data
fieldhand/weather.py    National Weather Service client
fieldhand/trace.py      trace capture and JSON/Markdown/HTML export
fieldhand/dryrun.py     scripted stand-in model for offline tests
app.py                  the public Streamlit app
run_demo.py             one run from the command line
evals/cases.json        14 evaluation cases
evals/run_eval.py       the harness (pass@1, pass^k)
tests/test_agent.py     offline tests
```

## Troubleshooting

- **Model not found (404):** set `FIELDHAND_MODEL` to a current ID from https://ai.google.dev/gemini-api/docs/models.
- **Rate limit (429) or overloaded (503):** the app falls back to `FIELDHAND_FALLBACK_MODEL`. The harness
  retries infrastructure errors and does not count them; if it still stops, rerun with `--resume`.
- **Thought summaries cause an error:** set `FIELDHAND_SHOW_THOUGHTS=false`.
- **No key yet:** pick "Dry run" in the app, or `--dry-run` on the command line. Dry runs are a scripted
  stand-in, labeled as such, and are not a run of the agent.

## AI use

Code and tests were written with Claude (Anthropic) as a coding assistant from the project design.
See the Milestone II report for the full disclosure.
