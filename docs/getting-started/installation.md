# Installation

This guide covers the recommended install flow for ClawVault on Linux (Ubuntu), macOS, Windows, and WSL.

## System requirements

- Node.js 18+ (Node.js 22+ recommended)
- npm 9+
- Supported OS: Linux, macOS, Windows, WSL

## Install the CLI

```bash
npm install -g clawvault
```

`qmd` is optional. ClawVault ships with an in-process BM25 search engine by default. Install `qmd` only if you want qmd fallback behavior:

```bash
bun install -g github:tobi/qmd
```

## Quick verification

After installation, run:

```bash
clawvault doctor
```

This checks your Node/npm environment, vault/config health, search setup, OpenClaw integration, and common Linux permission issues.

## Linux (Ubuntu 22.04 / 24.04) setup

### 1) Install Node.js and npm

Use your preferred version manager (nvm/fnm/asdf) and install Node.js 22 LTS. Validate:

```bash
node -v
npm -v
```

### 2) Configure npm global prefix (if you hit EACCES)

If `npm install -g clawvault` fails with permissions errors:

```bash
npm config set prefix ~/.npm-global
```

### 3) Add npm global bin directory to PATH

Append this to `~/.bashrc` (or `~/.zshrc`):

```bash
export PATH="$HOME/.npm-global/bin:$PATH"
```

Reload your shell:

```bash
source ~/.bashrc
```

### 4) Re-run install and verify

```bash
npm install -g clawvault
clawvault doctor
```

## Troubleshooting quick fixes

- `clawvault: command not found`
  - Ensure your npm global bin directory is in PATH.
  - Run `npm config get prefix` and confirm `<prefix>/bin` is exported.
- Global install fails with `EACCES`
  - Set user-owned npm prefix: `npm config set prefix ~/.npm-global`
  - Re-open terminal and retry install.
- `qmd` not found
  - This is optional. In-process BM25 search still works.
  - Install qmd only if you need fallback compatibility paths.
- OpenClaw plugin not registered
  - Run: `openclaw hooks install clawvault && openclaw hooks enable clawvault`
  - Verify: `openclaw hooks list --verbose`
