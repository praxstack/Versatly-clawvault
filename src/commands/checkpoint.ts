/**
 * Quick checkpoint command - fast state save for context death resilience
 */

import * as fs from 'fs';
import * as path from 'path';

export interface CheckpointOptions {
  workingOn?: string;
  focus?: string;
  blocked?: string;
  vaultPath: string;
}

export interface CheckpointData {
  timestamp: string;
  workingOn: string | null;
  focus: string | null;
  blocked: string | null;
  sessionId?: string;
}

const CLAWVAULT_DIR = '.clawvault';
const CHECKPOINT_FILE = 'last-checkpoint.json';
const SESSION_STATE_FILE = 'session-state.json';
const DIRTY_DEATH_FLAG = 'dirty-death.flag';

let pendingCheckpoint: NodeJS.Timeout | null = null;
let pendingData: { dir: string; data: CheckpointData } | null = null;

function ensureClawvaultDir(vaultPath: string): string {
  const dir = path.join(vaultPath, CLAWVAULT_DIR);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
  return dir;
}

function writeCheckpointToDisk(dir: string, data: CheckpointData): void {
  const checkpointPath = path.join(dir, CHECKPOINT_FILE);
  fs.writeFileSync(checkpointPath, JSON.stringify(data, null, 2));

  const flagPath = path.join(dir, DIRTY_DEATH_FLAG);
  fs.writeFileSync(flagPath, data.timestamp);
}

export async function flush(): Promise<CheckpointData | null> {
  if (pendingCheckpoint) {
    clearTimeout(pendingCheckpoint);
    pendingCheckpoint = null;
  }
  if (!pendingData) return null;

  const { dir, data } = pendingData;
  pendingData = null;
  writeCheckpointToDisk(dir, data);
  return data;
}

export async function checkpoint(options: CheckpointOptions): Promise<CheckpointData> {
  const dir = ensureClawvaultDir(options.vaultPath);
  
  const data: CheckpointData = {
    timestamp: new Date().toISOString(),
    workingOn: options.workingOn || null,
    focus: options.focus || null,
    blocked: options.blocked || null,
  };
  
  // Read session ID if available
  const sessionStatePath = path.join(dir, SESSION_STATE_FILE);
  if (fs.existsSync(sessionStatePath)) {
    try {
      const sessionState = JSON.parse(fs.readFileSync(sessionStatePath, 'utf-8'));
      data.sessionId = sessionState.sessionId;
    } catch {
      // Ignore parse errors
    }
  }

  // Debounce writes to avoid rapid write spam; last call wins.
  pendingData = { dir, data };
  if (pendingCheckpoint) clearTimeout(pendingCheckpoint);
  pendingCheckpoint = setTimeout(() => {
    void flush();
  }, 1000);

  return data;
}

export async function clearDirtyFlag(vaultPath: string): Promise<void> {
  const flagPath = path.join(vaultPath, CLAWVAULT_DIR, DIRTY_DEATH_FLAG);
  if (fs.existsSync(flagPath)) {
    fs.unlinkSync(flagPath);
  }
}

// Alias for CLI ergonomics (`clawvault clean-exit`)
export async function cleanExit(vaultPath: string): Promise<void> {
  await clearDirtyFlag(vaultPath);
}

export async function checkDirtyDeath(vaultPath: string): Promise<{
  died: boolean;
  checkpoint: CheckpointData | null;
  deathTime: string | null;
}> {
  const dir = path.join(vaultPath, CLAWVAULT_DIR);
  const flagPath = path.join(dir, DIRTY_DEATH_FLAG);
  const checkpointPath = path.join(dir, CHECKPOINT_FILE);
  
  if (!fs.existsSync(flagPath)) {
    return { died: false, checkpoint: null, deathTime: null };
  }
  
  const deathTime = fs.readFileSync(flagPath, 'utf-8').trim();
  
  let checkpoint: CheckpointData | null = null;
  if (fs.existsSync(checkpointPath)) {
    try {
      checkpoint = JSON.parse(fs.readFileSync(checkpointPath, 'utf-8'));
    } catch {
      // Ignore parse errors
    }
  }
  
  return { died: true, checkpoint, deathTime };
}

export async function setSessionState(vaultPath: string, sessionId: string): Promise<void> {
  const dir = ensureClawvaultDir(vaultPath);
  const sessionStatePath = path.join(dir, SESSION_STATE_FILE);
  
  fs.writeFileSync(sessionStatePath, JSON.stringify({
    sessionId,
    startedAt: new Date().toISOString(),
  }, null, 2));
}
