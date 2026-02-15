# ClawVault™ 🐘

**An elephant never forgets.**

Structured memory system for AI agents. Typed storage, knowledge graph, task management, and Obsidian-native dashboards — all local, all markdown.

[![npm](https://img.shields.io/npm/v/clawvault)](https://www.npmjs.com/package/clawvault) [![tests](https://img.shields.io/badge/tests-361%20passing-brightgreen)]()

🌐 [clawvault.dev](https://clawvault.dev) · 📚 [docs.clawvault.dev](https://docs.clawvault.dev) · 🛠️ [ClawHub Skill](https://clawhub.com/skills/clawvault)

> Works with [OpenClaw](https://openclaw.ai) agents or standalone. No cloud. No API keys. Just files.

## Install

```bash
npm install -g clawvault
```

## Quick Start

```bash
# Create a vault
clawvault init ~/memory --name my-brain

# Store memories by type
clawvault remember decision "Use PostgreSQL" --content "Chose for JSONB support"
clawvault remember lesson "Always checkpoint" --content "Context death is real"
clawvault capture "Quick thought to process later"

# Search
clawvault search "postgresql"           # Keyword (BM25)
clawvault vsearch "what database?"      # Semantic (local embeddings)

# Session lifecycle
clawvault wake                          # Start session, load context
clawvault sleep "built auth system" \   # End session with handoff
  --next "deploy to staging"

# Task management
clawvault task add "Ship v2" --owner agent --project acme --priority high
clawvault task list
clawvault blocked
clawvault backlog add "Explore caching" --project acme

# Visual dashboards (Obsidian JSON Canvas)
clawvault canvas --template brain           # System architecture overview
clawvault canvas --template project-board   # Task board by owner
```

## Features

### 📁 Typed Memory Storage
Every memory has a category: `decisions/`, `lessons/`, `people/`, `projects/`, `commitments/`. No more dumping everything into one file. "Show me all decisions" actually works.

### 🧠 Knowledge Graph
Wiki-links (`[[connections]]`) build a typed graph index. Query with graph-aware context that blends semantic search with relationship traversal.

```bash
clawvault graph                              # Graph summary
clawvault context "database migration" \     # Graph-aware context
  --profile planning --budget 2000
```

### ✅ Task Management
Tasks and backlog stored as markdown with frontmatter. Agents and humans share the same system.

```bash
clawvault task add "Fix auth" --owner bot --priority critical
clawvault task update fix-auth --status blocked --blocked-by "api-key"
clawvault blocked                            # Triage blocked work
clawvault backlog promote explore-caching    # Backlog → active task
```

### 🎨 Obsidian Dashboards
Generate visual dashboards as [JSON Canvas](https://jsoncanvas.org) files:

| Template | Description |
|----------|-------------|
| `brain` | 4-quadrant architecture: vault structure, direction, agent workspace, knowledge graph |
| `project-board` | Owner-centric kanban with status columns, priority icons, agent/human cards |
| `default` | Two-column dashboard with activity metrics and task triage |
| `sprint` | Weekly focus with sprint metrics and open loops |

```bash
clawvault canvas --template brain
clawvault canvas --template project-board --owner my-agent
```

### 🔭 Observational Memory
Auto-compress conversations into prioritized observations. Critical items route to vault categories automatically.

```bash
clawvault observe --compress session.md      # One-shot compression
clawvault observe --active                   # Incremental from transcripts
clawvault observe --cron                     # Cron-safe one-shot summary + exit code
```

### 🛡️ Context Death Recovery
Checkpoint/recover system keeps agents alive across crashes and session resets.

```bash
clawvault checkpoint --working-on "migration" --focus "step 3"
clawvault recover                            # After crash/reset
```

### 🌐 Tailscale Networking
Multi-vault collaboration over Tailscale with trust levels and cross-vault search.

```bash
clawvault serve                              # Start API server
clawvault peers                              # Manage vault peers
clawvault net-search "query"                 # Search across vaults
```

## Setup & Customization

```bash
# Full setup with neural graph theme + Obsidian Bases views
clawvault setup --theme neural --canvas brain

# Minimal agent vault
clawvault init ./memory --minimal

# Custom categories
clawvault init ./memory --categories "notes,research,code"

# Skip visual config
clawvault setup --no-graph-colors --no-bases
```

### Graph Themes

| Theme | Description |
|-------|-------------|
| `neural` | Dark background, colored nodes by category, green network links, golden glow |
| `minimal` | Subtle category colors, no background changes |
| `none` | Skip graph theming |

## Vault Structure

```
memory/
├── .clawvault.json          # Vault config
├── .clawvault/
│   └── graph-index.json     # Knowledge graph
├── tasks/                   # Active tasks (markdown + frontmatter)
├── backlog/                 # Ideas and future work
├── decisions/               # Choices with reasoning
├── lessons/                 # Things learned
├── people/                  # One file per person
├── projects/                # Active work
├── commitments/             # Promises and deadlines
├── inbox/                   # Quick captures
├── handoffs/                # Session continuity
├── ledger/
│   ├── raw/                 # Raw session transcripts
│   ├── observations/        # Compressed observations
│   └── reflections/         # Weekly reflections
├── all-tasks.base           # Obsidian Bases view
├── by-owner.base            # Tasks by owner
└── dashboard.canvas         # Generated dashboard
```

## For OpenClaw Agents

```bash
# Install as a skill
clawhub install clawvault

# Or add to your agent's tools
npm install -g clawvault
```

Add to your `AGENTS.md`:

```markdown
## Memory
- `clawvault wake` on session start
- `clawvault sleep "summary" --next "next steps"` on session end
- `clawvault checkpoint` every 10-15 min during heavy work
- `clawvault remember <type> "title" --content "..."` for important items
- `clawvault search "query"` before asking questions
```

## Requirements

- **Node.js 18+**
- **[qmd](https://github.com/Versatly/qmd)** — Local semantic search (optional but recommended)

## Docs

Full documentation at **[docs.clawvault.dev](https://docs.clawvault.dev)**

## License

MIT

---

*Built by [Versatly](https://versatly.com) — autonomous AI employees for businesses.* 🐘
