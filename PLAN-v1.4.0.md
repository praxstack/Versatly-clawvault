# ClawVault v1.4.0 Update Plan

*Created: 2026-02-03*
*Strategic pivot based on OpenClaw v2026.2.2 QMD integration*

---

## Executive Summary

OpenClaw v2026.2.2 added native QMD backend for memory. We **embrace this** by:
1. Making qmd a **hard dependency** (not optional)
2. Doubling down on **context death resilience** (checkpoint/recover/handoff)
3. Enhancing **template strategies** 
4. Improving **wiki-linking** for knowledge graph building
5. Integrating with OpenClaw's memory_search while keeping qmd as our core

---

## Part 1: Make QMD a Hard Dependency

### Changes to package.json
```json
{
  "dependencies": {
    "qmd": "github:tobi/qmd"  // Move from optional to required
  },
  // Remove from optionalDependencies, peerDependencies, peerDependenciesMeta
}
```

### Changes to Code
- Remove all `try/catch` wrappers for qmd imports
- Fail fast if qmd is not available
- Add clear error message: "ClawVault requires qmd. Install: bun install -g github:tobi/qmd"

### Files to Modify
- `package.json` — dependency changes
- `src/lib/search.ts` (or wherever search is implemented) — remove optional handling
- `src/commands/search.ts` — remove fallback logic
- `src/commands/vsearch.ts` — remove fallback logic

---

## Part 2: Double Down on Context Death Resilience

### Enhance Checkpoint Command
- Add `--auto` flag for automatic checkpointing (integrate with heartbeat)
- Add `--interval <minutes>` for suggested checkpoint frequency
- Store more metadata: session key, model, token estimate
- Add `--urgent` flag that also triggers OpenClaw wake

### Enhance Recover Command  
- Better detection of context death scenarios
- Show diff between last checkpoint and current state
- Add `--verbose` for full context dump
- Integration point: trigger on session start automatically

### Enhance Handoff Command
- Add `--to <agent>` for multi-agent handoffs
- Add `--priority <low|medium|high>` 
- Generate formatted handoff document
- Add `--notify` to alert the receiving agent

### New Command: `clawvault status`
- Show current vault health
- Last checkpoint age
- Uncommitted changes
- qmd index status
- Memory file counts by category

### Files to Create/Modify
- `src/commands/checkpoint.ts` — enhance with new flags
- `src/commands/recover.ts` — enhance with verbose/diff
- `src/commands/handoff.ts` — enhance with recipient/priority
- `src/commands/status.ts` — NEW: vault health check
- `src/lib/context-death.ts` — NEW: detection logic

---

## Part 3: Template Strategies

### New Command: `clawvault template`

```bash
# List available templates
clawvault template list

# Create from template
clawvault template create daily-note
clawvault template create project --name "New Project"
clawvault template create person --name "John Doe"

# Custom templates
clawvault template add ~/my-template.md --name "meeting-notes"
```

### Built-in Templates
1. **daily-note.md** — Daily journal with sections
2. **project.md** — Project tracking with status, tasks, links
3. **person.md** — Contact/relationship with context
4. **decision.md** — Decision log with reasoning, alternatives, outcome
5. **lesson.md** — Lesson learned with context, insight, application
6. **handoff.md** — Session handoff with state, next steps, blockers
7. **checkpoint.md** — Quick state capture
8. **meeting.md** — Meeting notes with attendees, agenda, actions

### Template Variables
```markdown
---
title: {{title}}
date: {{date}}
type: {{type}}
---

# {{title}}

Created: {{datetime}}
Author: {{agent_name}}
```

### Files to Create
- `src/commands/template.ts` — NEW: template management
- `templates/` — Update existing templates
- `src/lib/template-engine.ts` — NEW: variable interpolation

---

## Part 4: Wiki-Linking Enhancement

### Enhance `clawvault link` Command

```bash
# Link all files (existing)
clawvault link --all

# Link with auto-discovery of entities
clawvault link --auto-discover

# Create entity index from vault
clawvault link --rebuild-index

# Show orphan links (links to non-existent files)
clawvault link --orphans

# Show backlinks for a file
clawvault link --backlinks memory/2024-01-15.md
```

### Auto-Discovery Rules
- Detect capitalized proper nouns as potential entities
- Match against existing entities in vault
- Suggest new entities to create
- Learn from user confirmations

### Entity Index
- Central registry of all entities in vault
- Categories: people, projects, decisions, lessons, etc.
- Aliases support (John Doe → @johnd → [[people/john-doe]])

### Backlinks Feature
- Track which files link to which
- Generate backlinks section automatically
- Show in `clawvault status`

### Files to Modify/Create
- `src/commands/link.ts` — enhance with new features
- `src/lib/auto-linker.ts` — enhance auto-discovery
- `src/lib/entity-index.ts` — enhance with backlinks
- `src/lib/backlinks.ts` — NEW: backlink tracking

---

## Part 5: OpenClaw Integration

### Memory Search Bridge
When running inside OpenClaw:
- Detect OpenClaw environment (`OPENCLAW_SESSION_KEY` or similar)
- Write to OpenClaw's watched directories
- Let OpenClaw's memory_search index our files
- Keep qmd as our own search backend

### Hooks Integration
- Create HOOK.md for OpenClaw discovery
- Implement `agent:context_overflow` hook for emergency checkpoint
- Implement `agent:session_start` hook for auto-recover

### Files to Create
- `HOOK.md` — OpenClaw hook manifest
- `src/hooks/context-overflow.ts` — Emergency checkpoint
- `src/hooks/session-start.ts` — Auto-recover

---

## Implementation Order

### Phase 1: Core (Must Have)
1. [ ] Make qmd hard dependency
2. [ ] Enhance checkpoint with `--urgent` and metadata
3. [ ] Enhance recover with `--verbose` and diff
4. [ ] Add `clawvault status` command

### Phase 2: Templates
5. [ ] Create template command
6. [ ] Add built-in templates
7. [ ] Template variable interpolation

### Phase 3: Wiki-Linking
8. [ ] Add backlinks tracking
9. [ ] Add orphan link detection
10. [ ] Auto-discovery improvements

### Phase 4: OpenClaw Integration
11. [ ] Create HOOK.md
12. [ ] Implement context_overflow hook
13. [ ] Implement session_start hook

---

## Testing Checklist

- [ ] qmd install fails gracefully with clear error
- [ ] checkpoint creates valid JSON with all metadata
- [ ] recover detects context death scenarios
- [ ] templates interpolate variables correctly
- [ ] wiki-links resolve across categories
- [ ] backlinks update on file changes
- [ ] hooks trigger correctly in OpenClaw

---

## Version Bump

- Current: 1.3.1
- Target: 1.4.0 (minor bump for new features, no breaking changes)

---

## Success Criteria

1. **qmd is required** — Clear error if missing
2. **Context death is survivable** — Checkpoint/recover workflow is smooth
3. **Templates are useful** — Agents can quickly create structured notes
4. **Wiki-links build knowledge graph** — Backlinks, orphans, auto-discovery
5. **OpenClaw integration is seamless** — Hooks work, memory_search indexes our files

---

*This plan will be executed by GPT-5.2 Codex High via Cursor CLI*
