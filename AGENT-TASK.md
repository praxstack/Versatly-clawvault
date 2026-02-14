# Canvas Template System for ClawVault CLI

## Overview
Extend `clawvault canvas` to support templates. Currently it generates a single hardcoded dashboard layout. Add a `--template` flag to select from built-in templates, and a template engine that makes it easy to add new ones.

## What Exists
- `src/commands/canvas.ts` — generates `dashboard.canvas` with hardcoded layout
- `src/lib/canvas-layout.ts` — layout constants, node/group builders, JSON Canvas types
- `src/lib/vault-stats.ts` — collects vault statistics
- `src/lib/task-utils.ts` — task CRUD
- `src/lib/memory-graph.ts` — graph index loading

## What to Build

### 1. Template Engine (`src/lib/canvas-templates.ts`)

Define a `CanvasTemplate` interface:
```typescript
interface CanvasTemplate {
  id: string;
  name: string;
  description: string;
  generate(vaultPath: string, options: CanvasTemplateOptions): Canvas;
}

interface CanvasTemplateOptions {
  project?: string;      // filter by project
  dateRange?: { from: string; to: string };
  width?: number;
  height?: number;
}
```

Registry pattern: `registerTemplate(template)`, `getTemplate(id)`, `listTemplates()`.

### 2. Built-in Templates

**a) `default`** — current dashboard (move existing `generateCanvas` logic here)

**b) `project-board`** — Kanban-style project view
- 4 column groups: Open | In Progress | Blocked | Done
- File nodes for each task (linking to task .md files)
- Filter by `--project` flag
- Color-coded by priority

**c) `brain`** — Knowledge graph overview
- Center group: vault name + key stats
- Radial groups for top 6 categories (by file count)
- File nodes for top 5 entities per category
- Edges between entities that share wiki-links

**d) `sprint`** — Sprint/weekly focus
- Top row: active task count, blocked count, completion %
- Decision list from last 7 days (file nodes to decision .md files)
- Open loops: tasks open >7 days
- Recent observations summary

### 3. Update `src/commands/canvas.ts`

Add `--template` option:
```bash
clawvault canvas                              # default template
clawvault canvas --template project-board     # kanban view
clawvault canvas --template brain             # graph overview
clawvault canvas --template sprint            # weekly focus
clawvault canvas --template project-board --project clawvault  # filtered
clawvault canvas --list-templates             # show available templates
```

### 4. Tests

Add `src/lib/canvas-templates.test.ts`:
- Each template generates valid canvas JSON (has nodes array, edges array)
- Project filter works
- Template registry works (register, get, list)
- Default template matches current output

## Constraints
- Zero new dependencies
- All existing tests must pass
- Canvas output must be valid JSON Canvas spec 1.0
- Don't modify canvas-layout.ts types — extend if needed
- TypeScript strict mode

## Build & Test
```bash
npm run build
npm test
```
