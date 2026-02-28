import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import {
  backupSession,
  findMainSession,
  findSessionById,
  getOpenClawAgentsDir,
  getOpenClawDir,
  getSessionFilePath,
  getSessionsDir,
  getSessionsJsonPath,
  listAgents,
  listSessions,
  loadSessionsStore,
} from './session-utils.js';

function makeTempDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'clawvault-session-utils-'));
}

function writeJson(filePath: string, value: unknown): void {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, JSON.stringify(value, null, 2), 'utf-8');
}

describe('session-utils', () => {
  let tempDir: string;
  let originalOpenClawHome: string | undefined;
  let originalOpenClawStateDir: string | undefined;

  beforeEach(() => {
    tempDir = makeTempDir();
    originalOpenClawHome = process.env.OPENCLAW_HOME;
    originalOpenClawStateDir = process.env.OPENCLAW_STATE_DIR;
    delete process.env.OPENCLAW_HOME;
    delete process.env.OPENCLAW_STATE_DIR;
  });

  afterEach(() => {
    if (originalOpenClawHome === undefined) {
      delete process.env.OPENCLAW_HOME;
    } else {
      process.env.OPENCLAW_HOME = originalOpenClawHome;
    }

    if (originalOpenClawStateDir === undefined) {
      delete process.env.OPENCLAW_STATE_DIR;
    } else {
      process.env.OPENCLAW_STATE_DIR = originalOpenClawStateDir;
    }

    fs.rmSync(tempDir, { recursive: true, force: true });
  });

  it('resolves OpenClaw home and state directories from env vars', () => {
    const customHome = path.join(tempDir, 'custom-home');
    const customState = path.join(tempDir, 'custom-state');

    process.env.OPENCLAW_HOME = customHome;
    expect(getOpenClawDir()).toBe(customHome);
    expect(getOpenClawAgentsDir()).toBe(path.join(customHome, 'agents'));

    process.env.OPENCLAW_STATE_DIR = customState;
    expect(getOpenClawAgentsDir()).toBe(path.join(customState, 'agents'));

    process.env.OPENCLAW_HOME = '   ';
    delete process.env.OPENCLAW_STATE_DIR;
    expect(getOpenClawDir()).toBe(path.join(os.homedir(), '.openclaw'));
  });

  it('discovers only agents with a sessions directory', () => {
    process.env.OPENCLAW_STATE_DIR = path.join(tempDir, 'state');
    const agentsDir = getOpenClawAgentsDir();

    fs.mkdirSync(path.join(agentsDir, 'alice', 'sessions'), { recursive: true });
    fs.mkdirSync(path.join(agentsDir, 'bob'), { recursive: true });
    fs.mkdirSync(path.join(agentsDir, 'charlie'), { recursive: true });
    fs.writeFileSync(path.join(agentsDir, 'charlie', 'sessions'), 'not-a-dir', 'utf-8');

    expect(listAgents()).toEqual(['alice']);
  });

  it('loads sessions store JSON and handles invalid stores gracefully', () => {
    process.env.OPENCLAW_STATE_DIR = path.join(tempDir, 'state');

    writeJson(getSessionsJsonPath('alice'), {
      'agent:alice:main': { sessionId: 'sess-main', updatedAt: 100 },
    });
    fs.mkdirSync(getSessionsDir('bob'), { recursive: true });
    fs.writeFileSync(getSessionsJsonPath('bob'), '{bad-json}', 'utf-8');

    expect(loadSessionsStore('alice')).toEqual({
      'agent:alice:main': { sessionId: 'sess-main', updatedAt: 100 },
    });
    expect(loadSessionsStore('bob')).toBeNull();
    expect(loadSessionsStore('missing')).toBeNull();
  });

  it('finds main sessions and sessions by id with fallback keys', () => {
    process.env.OPENCLAW_STATE_DIR = path.join(tempDir, 'state');
    const agentId = 'alice';

    writeJson(getSessionsJsonPath(agentId), {
      [`agent:${agentId}:main`]: { sessionId: 'sess-main', updatedAt: 500 },
      [`agent:${agentId}:secondary`]: { sessionId: 'sess-2', updatedAt: 400 },
    });
    fs.mkdirSync(getSessionsDir(agentId), { recursive: true });
    fs.writeFileSync(getSessionFilePath(agentId, 'sess-main'), '{}\n', 'utf-8');
    fs.writeFileSync(getSessionFilePath(agentId, 'sess-2'), '{}\n', 'utf-8');
    fs.writeFileSync(getSessionFilePath(agentId, 'sess-lone'), '{}\n', 'utf-8');

    const main = findMainSession(agentId);
    expect(main).toMatchObject({
      sessionId: 'sess-main',
      sessionKey: `agent:${agentId}:main`,
      updatedAt: 500,
    });

    const known = findSessionById(agentId, 'sess-2');
    expect(known).toMatchObject({
      sessionId: 'sess-2',
      sessionKey: `agent:${agentId}:secondary`,
      updatedAt: 400,
    });

    const unknown = findSessionById(agentId, 'sess-lone');
    expect(unknown).toMatchObject({
      sessionId: 'sess-lone',
      sessionKey: `agent:${agentId}:unknown`,
    });

    fs.unlinkSync(getSessionFilePath(agentId, 'sess-main'));
    expect(findMainSession(agentId)).toBeNull();
  });

  it('lists sessions sorted by updatedAt and ignores backup/deleted/corrupted files', () => {
    process.env.OPENCLAW_STATE_DIR = path.join(tempDir, 'state');
    const agentId = 'alice';
    fs.mkdirSync(getSessionsDir(agentId), { recursive: true });

    fs.writeFileSync(getSessionFilePath(agentId, 'a'), '{}\n', 'utf-8');
    fs.writeFileSync(getSessionFilePath(agentId, 'b'), '{}\n', 'utf-8');
    fs.writeFileSync(path.join(getSessionsDir(agentId), 'c.jsonl.backup-20260214'), '{}\n', 'utf-8');
    fs.writeFileSync(path.join(getSessionsDir(agentId), 'd.deleted.jsonl'), '{}\n', 'utf-8');
    fs.writeFileSync(path.join(getSessionsDir(agentId), 'e.corrupted.jsonl'), '{}\n', 'utf-8');

    writeJson(getSessionsJsonPath(agentId), {
      [`agent:${agentId}:first`]: { sessionId: 'a', updatedAt: 100 },
      [`agent:${agentId}:second`]: { sessionId: 'b', updatedAt: 200 },
    });

    const sessions = listSessions(agentId);
    expect(sessions.map((session) => session.sessionId)).toEqual(['b', 'a']);
    expect(sessions.map((session) => session.sessionKey)).toEqual([
      `agent:${agentId}:second`,
      `agent:${agentId}:first`,
    ]);
  });

  it('creates timestamped backups for session files', () => {
    const source = path.join(tempDir, 'session.jsonl');
    fs.writeFileSync(source, '{"ok":true}\n', 'utf-8');

    const backupPath = backupSession(source);
    expect(backupPath).toContain('.backup-');
    expect(fs.existsSync(backupPath)).toBe(true);
    expect(fs.readFileSync(backupPath, 'utf-8')).toBe('{"ok":true}\n');
  });
});
