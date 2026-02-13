#!/usr/bin/env node

/**
 * ClawVault CLI 🐘
 * An elephant never forgets.
 */

import { Command } from 'commander';
import chalk from 'chalk';
import * as fs from 'fs';
import * as path from 'path';
import { spawn } from 'child_process';
import { registerMaintenanceCommands } from './register-maintenance-commands.js';
import { registerCoreCommands } from './register-core-commands.js';
import { registerQueryCommands } from './register-query-commands.js';
import { registerResilienceCommands } from './register-resilience-commands.js';
import { registerSessionLifecycleCommands } from './register-session-lifecycle-commands.js';
import { registerTemplateCommands } from './register-template-commands.js';
import { registerVaultOperationsCommands } from './register-vault-operations-commands.js';
import {
  ClawVault,
  createVault,
  findVault,
  QmdUnavailableError,
  QMD_INSTALL_COMMAND
} from '../dist/index.js';

const program = new Command();

const CLI_VERSION = (() => {
  try {
    const pkgUrl = new URL('../package.json', import.meta.url);
    const pkg = JSON.parse(fs.readFileSync(pkgUrl, 'utf-8'));
    return pkg.version || '0.0.0';
  } catch {
    return '0.0.0';
  }
})();

// Helper to get vault (required for most commands)
// Checks: 1) explicit path, 2) CLAWVAULT_PATH env, 3) walk up from cwd
async function getVault(vaultPath) {
  // Explicit path takes priority
  if (vaultPath) {
    const vault = new ClawVault(path.resolve(vaultPath));
    await vault.load();
    return vault;
  }
  
  // Check environment variable
  const envPath = process.env.CLAWVAULT_PATH;
  if (envPath) {
    const vault = new ClawVault(path.resolve(envPath));
    await vault.load();
    return vault;
  }
  
  // Walk up from cwd
  const vault = await findVault();
  if (!vault) {
    console.error(chalk.red('Error: No ClawVault found. Run `clawvault init` first.'));
    console.log(chalk.dim('Tip: Set CLAWVAULT_PATH environment variable to your vault path'));
    process.exit(1);
  }
  return vault;
}

function resolveVaultPath(vaultPath) {
  if (vaultPath) {
    return path.resolve(vaultPath);
  }
  if (process.env.CLAWVAULT_PATH) {
    return path.resolve(process.env.CLAWVAULT_PATH);
  }
  let current = process.cwd();
  while (true) {
    if (fs.existsSync(path.join(current, '.clawvault.json'))) {
      return current;
    }
    const parent = path.dirname(current);
    if (parent === current) {
      console.error(chalk.red('Error: No ClawVault found. Run `clawvault init` first.'));
      console.log(chalk.dim('Tip: Set CLAWVAULT_PATH environment variable to your vault path'));
      process.exit(1);
    }
    current = parent;
  }
}

async function runQmd(args) {
  return new Promise((resolve, reject) => {
    const proc = spawn('qmd', args, { stdio: 'inherit' });
    proc.on('close', (code) => {
      if (code === 0) resolve();
      else reject(new Error(`qmd exited with code ${code}`));
    });
    proc.on('error', (err) => {
      if (err?.code === 'ENOENT') {
        reject(new QmdUnavailableError());
      } else {
        reject(err);
      }
    });
  });
}

function printQmdMissing() {
  console.error(chalk.red('Error: ClawVault requires qmd.'));
  console.log(chalk.dim(`Install: ${QMD_INSTALL_COMMAND}`));
}

program
  .name('clawvault')
  .description('🐘 An elephant never forgets. Structured memory for AI agents.')
  .version(CLI_VERSION);

registerCoreCommands(program, {
  chalk,
  path,
  fs,
  createVault,
  getVault,
  runQmd
});

registerQueryCommands(program, {
  chalk,
  getVault,
  resolveVaultPath,
  QmdUnavailableError,
  printQmdMissing
});

registerSessionLifecycleCommands(program, {
  chalk,
  resolveVaultPath,
  QmdUnavailableError,
  printQmdMissing,
  getVault,
  runQmd
});

registerTemplateCommands(program, { chalk });
registerMaintenanceCommands(program, { chalk });
registerResilienceCommands(program, { chalk, resolveVaultPath });
registerVaultOperationsCommands(program, {
  chalk,
  fs,
  getVault,
  runQmd,
  resolveVaultPath,
  path
});

// Parse and run
program.parse();