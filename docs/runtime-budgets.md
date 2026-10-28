# Runtime retrieval and task budgets

These settings apply to the Python runtime used by the web app and the Python hands
operator. Android's native agent and the desktop's DeepSeek Harness chat loop have
their own execution engines; these settings do not change those engines.

## Recall across the whole store

Keyword and embedding retrieval search all saved memories. The default 1,000-row
limit belongs to the listing API only. A fact older than those 1,000 rows remains a
retrieval candidate, and building a vector index no longer removes its valid vector.

Keyword retrieval reuses a token-to-memory index while the database is unchanged.
Edits, forgetting, restoring, and writes from another connection invalidate it.
Chinese bigrams and the existing rare-word ranking are retained. The first retrieval
after a change still builds the keyword index over the whole store; vector recall
still performs a linear similarity search. This change does not introduce a vector
database or a background embedding service.

## Configure a task

Add this to `config/config.toml`. Limits default to zero, which leaves them disabled.
The numbers here are example limits, not model pricing.

```toml
[agent]
max_context_tokens = 32768

[agent.budget]
max_total_tokens = 200000
max_seconds = 600
max_cost_usd = 0
input_usd_per_million = 0
output_usd_per_million = 0
```

`max_context_tokens` caps a conservative estimate of the main Agent's input request, including the
system prompt, tool schemas, reasoning history and pictures. It is separate from the
model's output limit, `llm.max_tokens`. Text is estimated as UTF-8 bytes, including
JSON overhead; each image adds 4,096 estimated tokens. This is portable and deliberately
conservative for text, not an exact tokenizer or a guarantee about a vision model's
context window. Choose a limit that leaves room for the configured output.

When a request is too large, copies of tool results keep a labelled head and tail,
then old exchanges are removed as complete groups. Current-task user requests and
system instructions are retained intact, as are tool-call arguments. A request that
still cannot fit stops with a context-budget explanation before contacting the model.
The stored conversation keeps the original text.

`max_total_tokens` accumulates input and output across model calls in one `run()`.
Requests reserve estimated input and reduce `max_tokens` to the remaining allowance.
Provider-reported usage replaces the estimate when available. Missing usage, failed
requests and cancelled streams reserve the estimated input and full output allowance.
Retries made by the Agent and the operator share the same task budget. A new turn
starts a new budget; nested phone and computer operators inherit the parent's budget.

`max_seconds` also includes tools and approval waits. Reaching it cancels work
cooperatively. A tool that ignores cancellation or performs blocking work cannot be
forcibly undone. Once a token or cost limit is reached, no further tool is started and
no paid wrap-up request is sent. A plain model answer that completed within an in-flight
request can still be returned.

For a cost limit, supply your own **USD per million tokens** rates. Use upper rates
covering every model the task may call, including the hands model. Cached input is
charged at the full configured input rate in this estimate. A nonzero cost limit with
both rates zero is rejected. Charges for provider-side retries hidden inside a client,
embedding calls, images, clips, and other paid tools are outside this accounting.
The budget limits application requests, not the provider's invoice.

Standalone operators use the same fields under `[gui.budget]`; inside a runtime Agent
turn, `[agent.budget]` takes precedence. Limits are configured in TOML, not exposed as
new controls in each app. Each runtime turn writes a `task_budget` audit entry, and
`MuseAgent.last_budget` exposes its counters and stop reason. Operator outcomes carry
the same counters for evaluation reports.

See [GUI regression](gui-regression.md) for the fixed tasks and report command.
