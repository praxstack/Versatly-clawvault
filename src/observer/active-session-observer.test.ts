import { afterEach, describe, expect, it } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import {
  getScaledObservationThresholdBytes,
  observeActiveSessions,
  parseSessionSourceLabel
} from './active-session-observer.js';

const originalNoLlm = process.env.CLAWVAULT_NO_LLM;

function makeTempDir(prefix: string): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), prefix));
}

function writeVault(root: string): string {
  const vaultPath = path.join(root, 'vault');
  fs.mkdirSync(vaultPath, { recursive: true });
  fs.writeFileSync(path.join(vaultPath, '.clawvault.json'), JSON.stringify({ name: 'test' }), 'utf-8');
  return vaultPath;
}

function writeSessions(root: string, lines: string[]): { sessionsDir: string; transcriptPath: string; sessionId: string } {
  const sessionsDir = path.join(root, 'sessions');
  fs.mkdirSync(sessionsDir, { recursive: true });
  const sessionId = 'test-session-001';
  fs.writeFileSync(
    path.join(sessionsDir, 'sessions.json'),
    JSON.stringify({
      'agent:clawdious:main': {
        sessionId,
        updatedAt: Date.now()
      }
    }),
    'utf-8'
  );
  const transcriptPath = path.join(sessionsDir, `${sessionId}.jsonl`);
  fs.writeFileSync(transcriptPath, `${lines.join('\n')}\n`, 'utf-8');
  return { sessionsDir, transcriptPath, sessionId };
}

function messageLine(role: 'user' | 'assistant', text: string): string {
  return JSON.stringify({
    type: 'message',
    message: {
      role,
      content: text
    }
  });
}

afterEach(() => {
  process.env.CLAWVAULT_NO_LLM = originalNoLlm;
});

describe('active-session-observer', () => {
  it('calculates scaled thresholds by transcript size', () => {
    expect(getScaledObservationThresholdBytes(500 * 1024)).toBe(50 * 1024);
    expect(getScaledObservationThresholdBytes(1 * 1024 * 1024)).toBe(150 * 1024);
    expect(getScaledObservationThresholdBytes(5 * 1024 * 1024)).toBe(150 * 1024);
    expect(getScaledObservationThresholdBytes(6 * 1024 * 1024)).toBe(300 * 1024);
  });

  it('parses session source labels from session keys', () => {
    expect(parseSessionSourceLabel('agent:clawdious:main')).toBe('main');
    expect(parseSessionSourceLabel('agent:clawdious:telegram:dm:5439689035')).toBe('telegram-dm');
    expect(parseSessionSourceLabel('agent:clawdious:discord:channel:1469107483128762499')).toBe('discord');
    expect(parseSessionSourceLabel('agent:clawdious:telegram:group:-5114657181')).toBe('telegram-group');
  });

  it('observes only new transcript deltas and updates per-session cursors', async () => {
    process.env.CLAWVAULT_NO_LLM = '1';
    const root = makeTempDir('clawvault-active-observe-');
    const vaultPath = writeVault(root);
    const { sessionsDir, transcriptPath, sessionId } = writeSessions(root, [
      messageLine('user', 'Need migration plan'),
      messageLine('assistant', 'Use phased rollout')
    ]);

    try {
      const firstRun = await observeActiveSessions({
        vaultPath,
        sessionsDir,
        agentId: 'clawdious',
        minNewBytes: 1,
        threshold: 1,
        reflectThreshold: 99999
      });

      expect(firstRun.candidateSessions).toBe(1);
      expect(firstRun.observedSessions).toBe(1);
      expect(firstRun.cursorUpdates).toBe(1);

      const cursorPath = path.join(vaultPath, '.clawvault', 'observe-cursors.json');
      const cursors = JSON.parse(fs.readFileSync(cursorPath, 'utf-8')) as Record<string, {
        lastObservedOffset: number;
      }>;
      expect(cursors[sessionId]?.lastObservedOffset).toBeGreaterThan(0);

      const observationsDir = path.join(vaultPath, 'observations');
      const observationFiles = fs.readdirSync(observationsDir).filter((name) => name.endsWith('.md'));
      expect(observationFiles.length).toBeGreaterThan(0);
      const observationContent = fs.readFileSync(path.join(observationsDir, observationFiles[0]), 'utf-8');
      expect(observationContent).toContain('[main]');

      const secondRun = await observeActiveSessions({
        vaultPath,
        sessionsDir,
        agentId: 'clawdious',
        minNewBytes: 1,
        threshold: 1,
        reflectThreshold: 99999
      });
      expect(secondRun.candidateSessions).toBe(0);

      fs.appendFileSync(transcriptPath, `${messageLine('assistant', 'Added rollback checklist')}\n`, 'utf-8');

      const thirdRun = await observeActiveSessions({
        vaultPath,
        sessionsDir,
        agentId: 'clawdious',
        minNewBytes: 1,
        threshold: 1,
        reflectThreshold: 99999
      });
      expect(thirdRun.candidateSessions).toBe(1);
      expect(thirdRun.observedSessions).toBe(1);
    } finally {
      fs.rmSync(root, { recursive: true, force: true });
    }
  });

  it('supports dry-run without mutating cursor state', async () => {
    process.env.CLAWVAULT_NO_LLM = '1';
    const root = makeTempDir('clawvault-active-observe-dry-');
    const vaultPath = writeVault(root);
    const { sessionsDir } = writeSessions(root, [messageLine('user', 'draft one')]);

    try {
      const result = await observeActiveSessions({
        vaultPath,
        sessionsDir,
        agentId: 'clawdious',
        minNewBytes: 1,
        dryRun: true
      });

      expect(result.candidateSessions).toBe(1);
      expect(result.observedSessions).toBe(0);
      expect(result.cursorUpdates).toBe(0);

      const cursorPath = path.join(vaultPath, '.clawvault', 'observe-cursors.json');
      expect(fs.existsSync(cursorPath)).toBe(false);
    } finally {
      fs.rmSync(root, { recursive: true, force: true });
    }
  });
});
