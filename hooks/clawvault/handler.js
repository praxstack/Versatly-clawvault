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
function findVaultPath() {
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
function runClawvault(args) {
  try {
    const output = execSync(`clawvault ${args.join(' ')}`, {
      encoding: 'utf-8',
      timeout: 15000,
      stdio: ['pipe', 'pipe', 'pipe']
    });
    return { success: true, output: output.trim(), code: 0 };
  } catch (err) {
    return { 
      success: false, 
      output: err.stderr?.toString() || err.message || String(err),
      code: err.status || 1
    };
  }
}

// Handle gateway startup - check for context death
async function handleStartup(event) {
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
  
  if (output.includes('Context death detected') || output.includes('died') || output.includes('⚠️')) {
    // Extract relevant info
    const lines = output.split('\n');
    const workingOnLine = lines.find(l => l.toLowerCase().includes('working on'));
    const workingOn = workingOnLine ? workingOnLine.split(':').slice(1).join(':').trim() : null;
    
    const alertMsg = [
      '⚠️ **Context Death Detected**',
      workingOn ? `Last working on: ${workingOn}` : null,
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
async function handleNew(event) {
  const vaultPath = findVaultPath();
  if (!vaultPath) {
    console.log('[clawvault] No vault found, skipping auto-checkpoint');
    return;
  }

  // Build checkpoint info from event context
  const sessionKey = event.sessionKey || 'unknown';
  const source = event.context?.commandSource || 'cli';
  const workingOn = `Session reset via /new from ${source}`;

  console.log(`[clawvault] Auto-checkpoint before /new (session: ${sessionKey})`);

  const result = runClawvault([
    'checkpoint',
    '--working-on', `"${workingOn}"`,
    '--focus', `"Pre-reset checkpoint, session: ${sessionKey}"`,
    '-v', vaultPath
  ]);

  if (result.success) {
    console.log('[clawvault] Auto-checkpoint created');
  } else {
    console.warn('[clawvault] Auto-checkpoint failed:', result.output);
  }
}

// Main handler - route events
const handler = async (event) => {
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
