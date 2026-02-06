# ClawVault OpenClaw Hook Integration

## Overview

Build an OpenClaw hook that integrates ClawVault's context death resilience directly into OpenClaw's event system. **The hook lives inside the clawvault npm package** and is installed via `openclaw hooks install`.

## Distribution Model

```
clawvault (npm package)
├── package.json          # Add openclaw.hooks entry
├── hooks/
│   └── clawvault/
│       ├── HOOK.md       # Hook metadata
│       └── handler.ts    # Handler (compiled to handler.js)
├── src/
├── dist/
└── bin/
```

User installation:
```bash
npm install -g clawvault           # Install CLI
openclaw hooks install clawvault   # Register hook with OpenClaw
openclaw hooks enable clawvault    # Enable it
```

---

## Task 1: Create Hook Directory in Package

### Location: `/home/frame/Projects/clawvault/hooks/clawvault/`

### HOOK.md

```markdown
---
name: clawvault
description: "Context death resilience - auto-checkpoint and recovery detection"
metadata:
  openclaw:
    emoji: "🐘"
    events: ["gateway:startup", "command:new"]
    requires:
      bins: ["clawvault"]
---

# ClawVault Hook

Integrates ClawVault's context death resilience into OpenClaw:

- **On gateway startup**: Checks for context death, alerts agent
- **On /new command**: Auto-checkpoints before session reset

## Installation

```bash
npm install -g clawvault
openclaw hooks install clawvault
openclaw hooks enable clawvault
```

## Requirements

- ClawVault CLI installed globally
- Vault initialized (`clawvault setup` or `CLAWVAULT_PATH` set)

## What It Does

### Gateway Startup
1. Runs `clawvault recover --clear`
2. If context death detected, injects warning into first agent turn
3. Clears dirty death flag for clean session start

### Command: /new  
1. Creates automatic checkpoint with session info
2. Captures state even if agent forgot to handoff
3. Ensures continuity across session resets

## No Configuration Needed

Just enable the hook. It auto-detects vault path via:
1. `CLAWVAULT_PATH` environment variable
2. Walking up from cwd to find `.clawvault.json`
```

### handler.ts

```typescript
/**
 * ClawVault OpenClaw Hook
 * 
 * Provides automatic context death resilience:
 * - gateway:startup → detect context death, inject recovery info
 * - command:new → auto-checkpoint before session reset
 */

import { execSync } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

// Find vault by walking up directories
function findVaultPath(): string | null {
  // Check env first
  if (process.env.CLAWVAULT_PATH) {
    return process.env.CLAWVAULT_PATH;
  }

  // Walk up from cwd
  let dir = process.cwd();
  const root = path.parse(dir).root;
  
  while (dir !== root) {
    const configPath = path.join(dir, '.clawvault.json');
    if (fs.existsSync(configPath)) {
      return dir;
    }
    // Also check memory/ subdirectory (OpenClaw convention)
    const memoryConfig = path.join(dir, 'memory', '.clawvault.json');
    if (fs.existsSync(memoryConfig)) {
      return path.join(dir, 'memory');
    }
    dir = path.dirname(dir);
  }
  
  return null;
}

// Run clawvault command
function runClawvault(args: string[]): { success: boolean; output: string; code: number } {
  try {
    const output = execSync(`clawvault ${args.join(' ')}`, {
      encoding: 'utf-8',
      timeout: 15000,
      stdio: ['pipe', 'pipe', 'pipe']
    });
    return { success: true, output: output.trim(), code: 0 };
  } catch (err: any) {
    return { 
      success: false, 
      output: err.stderr?.toString() || err.message || String(err),
      code: err.status || 1
    };
  }
}

// Handle gateway startup - check for context death
async function handleStartup(event: any): Promise<void> {
  const vaultPath = findVaultPath();
  if (!vaultPath) {
    console.log('[clawvault] No vault found, skipping recovery check');
    return;
  }

  console.log(`[clawvault] Checking for context death (vault: ${vaultPath})`);

  const result = runClawvault(['recover', '--clear', '-v', vaultPath]);
  
  if (!result.success) {
    console.warn('[clawvault] Recovery check failed:', result.output);
    return;
  }

  // Parse output to detect if there was a death
  const output = result.output;
  
  if (output.includes('Context death detected') || output.includes('died')) {
    // Extract relevant info and inject into agent's first turn
    const lines = output.split('\n');
    const workingOn = lines.find(l => l.includes('Working on:'))?.replace('Working on:', '').trim();
    const deathTime = lines.find(l => l.includes('Death time:'))?.replace('Death time:', '').trim();
    
    const alertMsg = [
      '⚠️ **Context Death Detected**',
      workingOn ? `Last working on: ${workingOn}` : null,
      deathTime ? `Died: ${deathTime}` : null,
      'Run `clawvault wake` for full recovery context.'
    ].filter(Boolean).join('\n');

    // Inject into event messages if available
    if (event.messages && Array.isArray(event.messages)) {
      event.messages.push(alertMsg);
    }
    
    console.warn('[clawvault] ⚠️ Context death detected, alert injected');
  } else {
    console.log('[clawvault] Clean startup - no context death');
  }
}

// Handle /new command - auto-checkpoint before reset
async function handleNew(event: any): Promise<void> {
  const vaultPath = findVaultPath();
  if (!vaultPath) {
    console.log('[clawvault] No vault found, skipping auto-checkpoint');
    return;
  }

  // Build checkpoint info from event context
  const sessionKey = event.sessionKey || 'unknown';
  const source = event.context?.commandSource || 'unknown';
  const workingOn = `Session reset via /new (${source})`;

  console.log(`[clawvault] Auto-checkpoint before /new (session: ${sessionKey})`);

  const result = runClawvault([
    'checkpoint',
    '--working-on', JSON.stringify(workingOn),
    '--focus', JSON.stringify(`Pre-reset checkpoint, session: ${sessionKey}`),
    '-v', vaultPath
  ]);

  if (result.success) {
    console.log('[clawvault] Auto-checkpoint created');
  } else {
    console.warn('[clawvault] Auto-checkpoint failed:', result.output);
  }
}

// Main handler - route events
const handler = async (event: any): Promise<void> => {
  try {
    if (event.type === 'gateway' && event.action === 'startup') {
      await handleStartup(event);
      return;
    }

    if (event.type === 'command' && event.action === 'new') {
      await handleNew(event);
      return;
    }
  } catch (err) {
    console.error('[clawvault] Hook error:', err);
  }
};

export default handler;
```

---

## Task 2: Update package.json

Add the `openclaw.hooks` field to `/home/frame/Projects/clawvault/package.json`:

```json
{
  "name": "clawvault",
  "version": "1.4.1",
  ...
  "openclaw": {
    "hooks": ["./hooks/clawvault"]
  },
  "files": [
    "dist",
    "bin", 
    "templates",
    "hooks"
  ]
}
```

**Important:** Add `"hooks"` to the `files` array so it's included in npm publish.

---

## Task 3: Add Build Step for Hook

The hook handler needs to be compiled to JS. Update `/home/frame/Projects/clawvault/package.json` build script:

```json
{
  "scripts": {
    "build": "tsup src/index.ts ... && tsc hooks/clawvault/handler.ts --outDir hooks/clawvault --module ESNext --target ES2022",
    "build:hook": "tsc hooks/clawvault/handler.ts --outDir hooks/clawvault --module ESNext --target ES2022 --moduleResolution node"
  }
}
```

Or simpler - just write handler.js directly (no TS compilation needed for hooks).

---

## Task 4: Update SKILL.md  

Location: `/home/frame/.openclaw/workspace/skills/clawvault/SKILL.md`

Add after "Quick Setup" section:

```markdown
## OpenClaw Integration (Recommended)

ClawVault includes an OpenClaw hook for automatic context death resilience:

```bash
# Install hook from clawvault package
openclaw hooks install clawvault
openclaw hooks enable clawvault
```

**What the hook handles automatically:**
- Detects context death on gateway startup
- Injects recovery alert into first agent turn
- Auto-checkpoints before `/new` resets session

**Manual commands still valuable for:**
- `clawvault wake` — Full recap with projects and handoffs
- `clawvault sleep` — Detailed handoff with decisions and blockers
- `clawvault checkpoint` — Explicit save during heavy work

The hook is your safety net. Manual commands give richer context.
```

---

## Task 5: Update AGENTS.md

Location: `/home/frame/.openclaw/workspace/AGENTS.md`

Simplify the session rituals since hooks handle the basics:

### Replace "Every Session" with:

```markdown
## Every Session

Before doing anything else:

1. Read `SOUL.md` — this is who you are
2. Read `USER.md` — this is who you're helping  
3. Read `memory/YYYY-MM-DD.md` (today + yesterday) for recent context
4. **If in MAIN SESSION**: Also read `MEMORY.md`

**ClawVault hook handles context death automatically.** If previous session died, you'll see an alert. For full context:

```bash
clawvault wake    # Recovery + recap + projects
```
```

### Simplify "Session End Ritual":

```markdown
### 🚪 Session End Ritual

The ClawVault hook auto-checkpoints on `/new`. For **detailed handoffs**:

```bash
clawvault sleep "what you accomplished" \
  --next "what to do next" \
  --blocked "blockers" \
  --feeling "energy state"
```

Use `sleep` for important context. Hook catches basics if you forget.
```

---

## Task 6: Build and Test

```bash
cd /home/frame/Projects/clawvault

# Build everything
npm run build

# Test hook is in package
ls -la hooks/clawvault/

# Link for local testing  
npm link

# Test hook installation
openclaw hooks install /home/frame/Projects/clawvault
openclaw hooks list
openclaw hooks enable clawvault
openclaw hooks info clawvault
```

---

## Task 7: Commit

```bash
cd /home/frame/Projects/clawvault
git add -A
git commit -m "Add OpenClaw hook for automatic context death resilience

- Hook detects context death on gateway:startup
- Hook auto-checkpoints before command:new  
- Distributed via npm package (openclaw hooks install clawvault)
- Updated docs for hook-first approach"

cd /home/frame/.openclaw/workspace
git add skills/clawvault/SKILL.md AGENTS.md
git commit -m "Update ClawVault docs for hook-based approach"
```

---

## Success Criteria

1. ✅ `hooks/clawvault/` directory exists in package
2. ✅ `package.json` has `openclaw.hooks` field
3. ✅ `hooks` in `files` array for npm publish
4. ✅ `openclaw hooks install clawvault` works
5. ✅ `openclaw hooks enable clawvault` succeeds
6. ✅ Gateway restart shows recovery check log
7. ✅ `/new` command creates auto-checkpoint
8. ✅ SKILL.md documents hook installation
9. ✅ AGENTS.md reflects simplified rituals
