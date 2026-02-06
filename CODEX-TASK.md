# ClawVault Improvement Task

## Context
ClawVault is a structured memory system for OpenClaw AI agents. It helps them survive "context death" (losing state between sessions) through checkpoints, handoffs, and structured storage.

**Problem:** Agents aren't following best practices consistently. They forget to:
- Run `clawvault recover` on session start
- Run `clawvault handoff` on session end
- Checkpoint during heavy work
- Use wiki-links in their notes

## Your Task

Improve ClawVault to **actively help agents follow best practices**. Make it harder to do the wrong thing.

### 1. Add `clawvault wake` Command (NEW)
A single command agents run at session start that does everything:
```bash
clawvault wake
```
Should:
- Run recover (check for context death, show last checkpoint)
- Run recap (show recent handoffs, active projects)
- Show a brief "you were working on X" summary
- Clear the dirty-death flag
- Return exit code 0 if healthy, 1 if there was a death to review

### 2. Add `clawvault sleep` Command (NEW)
A single command for session end:
```bash
clawvault sleep "what I was working on"
```
Should:
- Create a handoff automatically
- Prompt for --next and --blocked if not provided
- Show confirmation of what was saved
- Optionally commit to git if repo is dirty

### 3. Improve `clawvault doctor` 
Add checks for:
- Days since last handoff (warn if >1 day)
- Days since last checkpoint (warn if >1 day during active use)
- Orphan link count (warn if >20)
- Inbox items pending processing (warn if >5)
- Missing CLAWVAULT_PATH in shell config

### 4. Update SKILL.md
Update the skill documentation to:
- Lead with `wake` and `sleep` as the primary commands
- Add a "Quick Start for New Agents" section
- Include a checklist agents can copy into their AGENTS.md
- Add troubleshooting for common issues

### 5. Add Shell Integration Helper
```bash
clawvault shell-init
```
Should output shell code that:
- Sets CLAWVAULT_PATH if detectable
- Adds aliases: `cvwake`, `cvsleep`, `cvcheck`
- Can be added to ~/.bashrc

## Files to Modify

- `/home/frame/Projects/clawvault/src/commands/` - Add wake.ts, sleep.ts
- `/home/frame/Projects/clawvault/src/commands/doctor.ts` - Enhance checks
- `/home/frame/Projects/clawvault/bin/clawvault` - Register new commands
- `/home/frame/Projects/clawvault/SKILL.md` - Update documentation
- `/home/frame/Projects/clawvault/README.md` - Update with new commands

## Testing

After changes:
```bash
npm run build
npm link
clawvault wake
clawvault sleep "testing new commands"
clawvault doctor
```

## Style Guide

- TypeScript, match existing code style
- Use chalk for colored output
- Keep commands fast (<500ms for wake/sleep)
- Helpful error messages that tell agents what to do

## Success Criteria

1. `clawvault wake` works and shows useful context
2. `clawvault sleep` creates proper handoffs
3. `clawvault doctor` catches the issues we identified
4. SKILL.md is updated and clear
5. All existing tests still pass
