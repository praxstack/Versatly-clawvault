# ClawVault™ 🐘

**An elephant never forgets.**

Structured memory system for AI agents. Store, search, and link memories across sessions.

🌐 **Website:** [clawvault.dev](https://clawvault.dev) | 📦 **npm:** [clawvault](https://www.npmjs.com/package/clawvault) | 🛠️ **ClawHub:** [clawvault skill](https://clawhub.com/skills/clawvault)

> **Built for [OpenClaw](https://openclaw.ai)** — the AI agent framework. Works standalone too.

## Install for OpenClaw Agents

```bash
# Install the skill (recommended for OpenClaw agents)
clawhub install clawvault

# Or install the CLI globally
npm install -g clawvault
```

## Requirements

- **Node.js 18+**
- **[qmd](https://github.com/Versatly/qmd)** — Local semantic search (required)

```bash
# Install qmd first
bun install -g qmd   # or: npm install -g qmd

# Then install clawvault
npm install -g clawvault
```

## Blog & Resources

- **Blog:** [clawvault.dev/blog](https://clawvault.dev/blog/)
- **RSS:** [feed.xml](https://clawvault.dev/blog/feed.xml)
- **Sitemap:** [sitemap.xml](https://clawvault.dev/sitemap.xml)

## Why ClawVault?

AI agents forget things. Context windows overflow, sessions end, important details get lost. ClawVault fixes that:

- **Structured storage** — Organized categories, not random notes
- **Local search** — qmd provides BM25 + semantic search with local embeddings (no API quotas)
- **Wiki-links** — `[[connections]]` visible in Obsidian's graph view
- **Session continuity** — Handoff/recap system for context death
- **Token efficient** — Search instead of loading entire memory files

## Quick Start

```bash
# Initialize vault with qmd collection
clawvault init ~/memory --qmd-collection my-memory

# Store memories
clawvault remember decision "Use qmd" --content "Local embeddings, no API limits"
clawvault remember lesson "Context death is survivable" --content "Write it down"
clawvault capture "Quick note to process later"

# Search (uses qmd)
clawvault search "decision"           # BM25 keyword search
clawvault vsearch "what did I decide" # Semantic search

# Session management
clawvault wake
clawvault sleep "build wake/sleep commands" --next "run doctor"
clawvault handoff --working-on "task1" --next "task2"   # Manual handoff (advanced)
clawvault recap                                         # Manual recap (advanced)
```

**Tip:** Set `CLAWVAULT_PATH` to skip directory walk (or use `shell-init`):
```bash
echo 'export CLAWVAULT_PATH="$HOME/memory"' >> ~/.bashrc
eval "$(clawvault shell-init)"
```

## ClawVault Cloud

ClawVault Cloud extends local memory with org-linked decision traces. The local vault stays your source of truth, and cloud sync adds cross-agent visibility plus centralized audit trails.

- **Local-first writes** - `trace emit` always appends locally before sync attempts.
- **Queued and retryable sync** - traces are buffered and sent with `clawvault sync`.
- **Org-linked vault identity** - each vault can be linked once, then monitored with status checks.
- **Backwards-compatible sync command** - `clawvault sync <target>` still syncs vault files to a folder.

Quick setup:

```bash
# 1) Save your cloud API key
clawvault config --cloud-key cvk_xxx

# 2) Link this local vault to your org
clawvault org link --vault ~/memory

# 3) Verify cloud + org link state
clawvault org status

# 4) Emit trace events and sync queued traces
clawvault trace emit --summary "Approved 20% discount for ACME"
clawvault sync
```

## Search: qmd vs memory_search

**Use `qmd` (or `clawvault search`) — not `memory_search`**

| Tool | Backend | Speed | API Limits |
|------|---------|-------|------------|
| `qmd search` / `clawvault search` | Local BM25 | Instant | None |
| `qmd vsearch` / `clawvault vsearch` | Local embeddings | Fast | None |
| `memory_search` | Gemini API | Variable | **Yes, hits quotas** |

```bash
# ✅ Use this
qmd search "query" -c my-memory
clawvault search "query"

# ❌ Avoid (API quotas)
memory_search
```

## Vault Structure

```
my-memory/
├── .clawvault.json      # Config (includes qmd collection name)
├── decisions/           # Choices with reasoning
├── lessons/             # Things learned
├── people/              # One file per person
├── projects/            # Active work
├── commitments/         # Promises and deadlines
├── inbox/               # Quick capture (process later)
└── handoffs/            # Session continuity
```

## Commands

### Store Memories

```bash
# With type classification (recommended)
clawvault remember <type> <title> --content "..."
# Types: decision, lesson, fact, commitment, project, person

# Quick capture
clawvault capture "Note to self"

# Manual store
clawvault store -c decisions -t "Title" --content "..."
```

**Note:** All write commands auto-update the qmd index. Use `--no-index` to skip.

### Search

```bash
clawvault search "query"           # BM25 keyword
clawvault search "query" -c people # Filter by category
clawvault vsearch "query"          # Semantic (local embeddings)
```

### Browse

```bash
clawvault list                # All documents
clawvault list decisions      # By category
clawvault get decisions/title # Specific document
clawvault stats               # Vault overview
```

### Session Continuity

```bash
# Start a session (recover + recap + summary)
clawvault wake

# End a session with a handoff
clawvault sleep "building CRM, fixing webhook" \
  --blocked "waiting for API key" \
  --next "deploy to production" \
  --decisions "chose Supabase over Firebase" \
  --feeling "focused"

# Manual tools (advanced)
clawvault handoff --working-on "task1" --next "task2"
clawvault recap --brief   # Token-efficient recap

# Health check
clawvault doctor
```

### Cloud Sync

```bash
# Configure cloud API access
clawvault config --cloud-key cvk_xxx
clawvault config --cloud-api-url https://api.clawvault.dev # optional override

# Link local vault to cloud org/vault ID
clawvault org link --vault ~/memory
clawvault org status

# Emit decision traces (summary-only or JSON payload)
clawvault trace emit --summary "Approved 20% discount for ACME"
clawvault trace emit --trace-file ./trace.json

# Sync queued cloud traces (no target argument)
clawvault sync
clawvault sync --all
clawvault sync --limit 25

# Existing file sync still works when target is provided
clawvault sync ./obsidian   # existing file sync to target folder
```

## Agent Setup (AGENTS.md)

Add this to your `AGENTS.md` to ensure proper memory habits:

```markdown
## Memory

**Write everything down. Memory doesn't survive session restarts.**

### Search (use qmd, not memory_search)
\`\`\`bash
qmd search "query" -c your-memory    # Fast keyword
qmd vsearch "query" -c your-memory   # Semantic
\`\`\`

### Store
\`\`\`bash
clawvault remember decision "Title" --content "..."
clawvault remember lesson "Title" --content "..."
\`\`\`

### Session Start
\`\`\`bash
clawvault wake
\`\`\`

### Session End
\`\`\`bash
clawvault sleep "..." --next "..."
\`\`\`

### Checkpoint (during heavy work)
\`\`\`bash
clawvault checkpoint --working-on "..." --focus "..." --blocked "..."
\`\`\`

### Why qmd over memory_search?
- Local embeddings — no API quotas
- Always works — no external dependencies
- Fast — instant BM25, quick semantic
```

## Templates

ClawVault includes templates for common memory types:

- `decision.md` — Choices with context and reasoning
- `lesson.md` — Things learned
- `person.md` — People you work with
- `project.md` — Active work
- `handoff.md` — Session state before context death
- `daily.md` — Daily notes

Use with: `clawvault store -c category -t "Title" -f decision`

## Library Usage

```typescript
import { ClawVault, createVault, findVault } from 'clawvault';

const vault = await createVault('./memory', { qmdCollection: 'my-memory' });

await vault.store({
  category: 'decisions',
  title: 'Use ClawVault',
  content: 'Decided to use ClawVault for memory.',
});

const results = await vault.find('memory', { limit: 5 });
```

## License

MIT

---

*"An elephant never forgets." — Now neither do you.* 🐘
