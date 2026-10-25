# ARKEV's MiroFish

A fork of [nikmcfly/MiroFish-Offline](https://github.com/nikmcfly/MiroFish-Offline), which is
itself a fork of [666ghj/MiroFish](https://github.com/666ghj/MiroFish). It's the simulation
engine behind ARKEV's Idea Lab. The code is AGPL-3.0, so this fork is public and anyone using
the service can get the source here.

## What ARKEV changed

- **`claude-proxy/`**: an OpenAI-compatible `/v1/chat/completions` endpoint answered by the
  official Claude Code CLI (`claude -p`).
  - It runs on the owner's Claude subscription (`CLAUDE_CODE_OAUTH_TOKEN`, from
    `claude setup-token`) and never uses an API key.
  - It supports OpenAI **tool calls**. OASIS's simulated people act through tools (post, like,
    comment), so a text-only proxy would leave them unable to act.
  - Claude gets no tools of its own, no MCP and no settings.
  - Callers need `PROXY_API_KEY`.
  - At most `PROXY_MAX_CONCURRENT` calls run at once (default 3), and the rest wait in line.
  - It is adapted from JayFarei/MiroFish's text-only proxy.
- **`MIROFISH_API_TOKEN`**: every `/api/` route needs it as a Bearer token. MiroFish had no
  login at all.
- **CORS** is off unless `MIROFISH_CORS_ORIGINS` is set.
- **`Dockerfile.backend`**: the backend alone, on CPU torch, with no CUDA packages.
- **`docker-compose.coolify.yml`**: the backend, the Claude proxy, Neo4j and Ollama (embeddings
  only). Nothing is published; Core reaches it as `arkev-mirofish:5001` on the coolify network.
- **CI** (`.github/workflows/arkev.yml`): runs the proxy tests, builds both images and checks
  that the API refuses requests without the token.

## How a run works

1. Seed material is turned into an **ontology**, then a **graph**: entities and relationships
   in Neo4j, embedded with nomic-embed-text.
2. **Profiles** for the simulated people are generated from the graph.
3. An **OASIS** simulation runs: Twitter-like and Reddit-like feeds, where the people post,
   reply, like and follow over rounds.
4. A **report agent** writes the prediction, using graph tools.
5. You can **interview** any simulated person.

Each step is a call to `/api/graph/*`, `/api/simulation/*` or `/api/report/*`.

## Running it

In Coolify, set these four values, then deploy:

- `CLAUDE_CODE_OAUTH_TOKEN`
- `MIROFISH_API_TOKEN`
- `MIROFISH_PROXY_KEY`
- `MIROFISH_NEO4J_PASSWORD`

The server has no GPU. All chat goes to Claude through the proxy, and Ollama only makes
embeddings.

On the subscription, a run is limited by Claude's usage limits. Keep crowds small (tens of
people, a few rounds). A big run can use a large share of a 5-hour window.
