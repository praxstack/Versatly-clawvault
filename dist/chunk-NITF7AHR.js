// src/commands/checkpoint.ts
import * as fs from "fs";
import * as path from "path";
var CLAWVAULT_DIR = ".clawvault";
var CHECKPOINT_FILE = "last-checkpoint.json";
var SESSION_STATE_FILE = "session-state.json";
var DIRTY_DEATH_FLAG = "dirty-death.flag";
var pendingCheckpoint = null;
var pendingData = null;
function ensureClawvaultDir(vaultPath) {
  const dir = path.join(vaultPath, CLAWVAULT_DIR);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
  return dir;
}
function writeCheckpointToDisk(dir, data) {
  const checkpointPath = path.join(dir, CHECKPOINT_FILE);
  fs.writeFileSync(checkpointPath, JSON.stringify(data, null, 2));
  const flagPath = path.join(dir, DIRTY_DEATH_FLAG);
  fs.writeFileSync(flagPath, data.timestamp);
}
async function flush() {
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
async function checkpoint(options) {
  const dir = ensureClawvaultDir(options.vaultPath);
  const data = {
    timestamp: (/* @__PURE__ */ new Date()).toISOString(),
    workingOn: options.workingOn || null,
    focus: options.focus || null,
    blocked: options.blocked || null
  };
  const sessionStatePath = path.join(dir, SESSION_STATE_FILE);
  if (fs.existsSync(sessionStatePath)) {
    try {
      const sessionState = JSON.parse(fs.readFileSync(sessionStatePath, "utf-8"));
      data.sessionId = sessionState.sessionId;
    } catch {
    }
  }
  pendingData = { dir, data };
  if (pendingCheckpoint) clearTimeout(pendingCheckpoint);
  pendingCheckpoint = setTimeout(() => {
    void flush();
  }, 1e3);
  return data;
}
async function clearDirtyFlag(vaultPath) {
  const flagPath = path.join(vaultPath, CLAWVAULT_DIR, DIRTY_DEATH_FLAG);
  if (fs.existsSync(flagPath)) {
    fs.unlinkSync(flagPath);
  }
}
async function checkDirtyDeath(vaultPath) {
  const dir = path.join(vaultPath, CLAWVAULT_DIR);
  const flagPath = path.join(dir, DIRTY_DEATH_FLAG);
  const checkpointPath = path.join(dir, CHECKPOINT_FILE);
  if (!fs.existsSync(flagPath)) {
    return { died: false, checkpoint: null, deathTime: null };
  }
  const deathTime = fs.readFileSync(flagPath, "utf-8").trim();
  let checkpoint2 = null;
  if (fs.existsSync(checkpointPath)) {
    try {
      checkpoint2 = JSON.parse(fs.readFileSync(checkpointPath, "utf-8"));
    } catch {
    }
  }
  return { died: true, checkpoint: checkpoint2, deathTime };
}
async function setSessionState(vaultPath, sessionId) {
  const dir = ensureClawvaultDir(vaultPath);
  const sessionStatePath = path.join(dir, SESSION_STATE_FILE);
  fs.writeFileSync(sessionStatePath, JSON.stringify({
    sessionId,
    startedAt: (/* @__PURE__ */ new Date()).toISOString()
  }, null, 2));
}

export {
  flush,
  checkpoint,
  clearDirtyFlag,
  checkDirtyDeath,
  setSessionState
};
