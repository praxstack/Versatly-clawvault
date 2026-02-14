# ClawVault Tailscale Network Layer — Native Multi-Agent Collaboration

## Overview

Add a native Tailscale-based network layer to ClawVault that lets multiple agents on different machines (different OpenClaw gateways) collaborate through shared vault access over a Tailscale network. This is NOT federation (symlinks/copies) — this is live network access to vault data.

## Architecture

### Core Concept
Each ClawVault instance can optionally run a lightweight HTTP server on its Tailscale IP, exposing a read/write API for vault operations. Other agents on the same Tailnet can discover and connect to these vaults as "remote vaults".

### Components to Build

#### 1. `clawvault serve` command (NEW)
Starts a lightweight HTTP server bound to the Tailscale interface.

```bash
clawvault serve [--port 7283] [--vault .] [--read-only] [--auth-token <token>]
```

- Binds to Tailscale IP only (100.x.x.x) — never 0.0.0.0
- Default port: 7283 (CVLT on phone keypad)
- Optional `--read-only` mode for vaults that only publish
- Bearer token auth (auto-generated if not provided, stored in `.clawvault.json`)
- Serves these endpoints:

**API Endpoints:**
```
GET  /v1/status          — Vault name, version, stats, capabilities
GET  /v1/search?q=...    — Search vault (keyword)
GET  /v1/vsearch?q=...   — Semantic search (if qmd available)
GET  /v1/documents/:id   — Get a specific document by ID
GET  /v1/observations     — Get observations (date range params)
GET  /v1/graph            — Get memory graph (nodes + edges)
GET  /v1/context          — Get context-injected content (like `clawvault context`)
POST /v1/store            — Store a document (body: {category, title, content, frontmatter})
POST /v1/observe          — Submit an observation
GET  /v1/ledger/reflections — Get reflections
GET  /v1/ledger/observations/:date — Get compiled observations for a date
```

#### 2. Network discovery via `.clawvault.json` config

Add a `network` section to `.clawvault.json`:

```json
{
  "network": {
    "serve": {
      "enabled": true,
      "port": 7283,
      "readOnly": false,
      "authToken": "cv_xxxxx"
    },
    "peers": [
      {
        "name": "eli-vault",
        "host": "elis-mac-mini",
        "port": 7283,
        "authToken": "cv_yyyyy",
        "trust": "read"
      },
      {
        "name": "roman-vault",
        "host": "openclaw-hub",
        "port": 7283,
        "authToken": "cv_zzzzz",
        "trust": "read-write"
      }
    ]
  }
}
```

Trust levels:
- `read` — can search and read documents from this peer
- `read-write` — can also store documents and submit observations
- `full` — can also modify graph, run admin commands

#### 3. `clawvault peers` command (NEW)

```bash
clawvault peers                    # List configured peers + status (online/offline)
clawvault peers add <name> <host>  # Add a peer (interactive auth token exchange)
clawvault peers remove <name>      # Remove a peer
clawvault peers ping [name]        # Ping one or all peers
```

#### 4. `clawvault net-search` command (NEW)

Search across all connected peer vaults + local vault:

```bash
clawvault net-search "query"       # Search all peers + local
clawvault net-search "query" --peer eli-vault  # Search specific peer
clawvault net-search "query" --local-only      # Just local (same as regular search)
```

Results include source attribution: `[local] result...` vs `[eli-vault] result...`

#### 5. Cross-vault context injection

Extend `clawvault context` with `--include-peers` flag:

```bash
clawvault context --include-peers   # Include high-importance observations from peers
clawvault context --peer eli-vault  # Include context from specific peer
```

This fetches structural (i>=0.8) observations from peer vaults and includes them in the context output, attributed to source.

#### 6. Observation forwarding

When observing a session, if the observation mentions entities that belong to a peer vault, optionally forward the observation:

```json
{
  "network": {
    "forwarding": {
      "enabled": true,
      "rules": [
        { "entity": "hale-pet-door", "peer": "eli-vault" },
        { "tag": "#sales", "peer": "roman-vault" }
      ]
    }
  }
}
```

### Implementation Details

#### HTTP Server
- Use Node.js built-in `http` module (NO express, NO external HTTP framework)
- Keep it minimal — this is an agent tool, not a web app
- JSON request/response throughout
- Auth via `Authorization: Bearer cv_xxxxx` header
- Bind specifically to Tailscale interface IP (detect via `tailscale ip -4`)
- Graceful shutdown on SIGTERM/SIGINT

#### File Structure
```
src/
  network/
    server.ts          — HTTP server implementation
    server.test.ts     — Server tests
    client.ts          — Client for connecting to peer vaults  
    client.test.ts     — Client tests
    discovery.ts       — Peer discovery + Tailscale integration
    discovery.test.ts  — Discovery tests
    routes.ts          — API route handlers
    routes.test.ts     — Route tests
    auth.ts            — Token generation + validation
    auth.test.ts       — Auth tests
    types.ts           — Network-specific types
  commands/
    serve.ts           — `clawvault serve` command
    serve.test.ts
    peers.ts           — `clawvault peers` command  
    peers.test.ts
    net-search.ts      — `clawvault net-search` command
    net-search.test.ts
```

#### Tailscale Integration
- Detect Tailscale IP via `tailscale ip -4` (shell out)
- Detect Tailscale status via `tailscale status --json`
- Resolve peer hostnames via Tailscale MagicDNS (just use hostname directly)
- If Tailscale is not available, `serve` refuses to start (security: never bind to non-Tailscale interfaces)
- Auto-detect if running on Tailscale network

#### Security Model
- Server ONLY binds to Tailscale IP (100.x.x.x range)
- Bearer token auth on every request
- Trust levels limit what operations peers can perform
- `--read-only` mode for sensitive vaults
- No plaintext credentials in logs
- Token format: `cv_` prefix + 32 random hex chars

#### Registration in CLI
- Register `serve`, `peers`, and `net-search` commands in the CLI entry point
- Follow existing command registration pattern (see `bin/` directory for how commands are registered)
- All new commands should have `--help` with usage examples

### Testing Requirements
- Unit tests for server, client, routes, auth, discovery
- Integration test: start server -> client connects -> search -> get results
- Test auth rejection (wrong token, no token)
- Test read-only mode (reject writes)
- Test Tailscale detection (mock `tailscale` binary)
- Test peer discovery (mock config)
- Run `npm test` to verify — must pass ALL existing 353 tests plus new ones
- Run `npm run build` to verify TypeScript compilation

### Constraints
- Zero new runtime dependencies (use Node.js built-in `http`, `crypto`, `child_process`)
- TypeScript strict mode
- Follow existing code patterns in `src/lib/` and `src/commands/`
- ESM modules (the project uses `"type": "module"`)
- Don't modify existing tests
- Don't break any existing functionality
- Keep the server lightweight — this runs on resource-constrained machines (Raspberry Pi, Mac Mini)

### What Done Looks Like
1. `clawvault serve` starts an HTTP server on Tailscale IP
2. `clawvault peers add eli elis-mac-mini` configures a peer
3. `clawvault peers` shows online/offline status of all peers
4. `clawvault net-search "query"` returns results from local + all online peers
5. `clawvault context --include-peers` includes peer observations in context
6. All existing tests pass + new tests for network layer
7. `npm run build` succeeds

### Build & Test Commands
```bash
npm run build          # TypeScript compilation
npm test               # Run all tests (vitest)
npm run test:coverage  # Coverage report
```

### Reference Files
- Existing vault operations: `src/lib/vault.ts`
- Config handling: `src/lib/config.ts`
- Types: `src/types.ts`
- Command pattern: `src/commands/status.ts` (simple), `src/commands/observe.ts` (complex)
- CLI registration: `bin/` directory
- Search implementation: `src/lib/search.ts`
- Observation format: `src/lib/observation-format.ts`
- Ledger: `src/lib/ledger.ts`
- Memory graph: `src/lib/memory-graph.ts`
