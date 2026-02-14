import { afterEach, describe, expect, it, vi } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

const { parseSessionFileMock, processMessagesMock, flushMock, observeActiveSessionsMock } = vi.hoisted(() => ({
  parseSessionFileMock: vi.fn(),
  processMessagesMock: vi.fn(),
  flushMock: vi.fn(),
  observeActiveSessionsMock: vi.fn()
}));

vi.mock('../observer/session-parser.js', () => ({
  parseSessionFile: parseSessionFileMock
}));

vi.mock('../observer/observer.js', () => ({
  Observer: class {
    async processMessages(messages: string[]): Promise<void> {
      await processMessagesMock(messages);
    }

    async flush(): Promise<{ observations: string; routingSummary: string }> {
      return flushMock();
    }
  }
}));

vi.mock('../observer/watcher.js', () => ({
  SessionWatcher: class {
    async start(): Promise<void> {
      return;
    }

    async stop(): Promise<void> {
      return;
    }
  }
}));

vi.mock('../observer/active-session-observer.js', () => ({
  observeActiveSessions: observeActiveSessionsMock
}));

import { observeCommand } from './observe.js';

function makeTempFile(content: string): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'clawvault-observe-'));
  const filePath = path.join(dir, 'session.txt');
  fs.writeFileSync(filePath, content, 'utf-8');
  return filePath;
}

afterEach(() => {
  vi.clearAllMocks();
});

describe('observeCommand', () => {
  it('uses parseSessionFile for one-shot compression input', async () => {
    const sessionPath = makeTempFile('line one\nline two');
    parseSessionFileMock.mockReturnValue(['user: line one', 'assistant: line two']);
    processMessagesMock.mockResolvedValue(undefined);
    flushMock.mockResolvedValue({
      observations: '## 2026-02-11\n\n🟡 09:00 Parsed with session parser',
      routingSummary: 'Routed 1 observations → lessons: 1'
    });

    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined);

    await observeCommand({
      vaultPath: '/tmp/vault',
      compress: sessionPath,
      threshold: 10,
      reflectThreshold: 20
    });

    expect(parseSessionFileMock).toHaveBeenCalledWith(path.resolve(sessionPath));
    expect(processMessagesMock).toHaveBeenCalledWith(['user: line one', 'assistant: line two']);
    expect(flushMock).toHaveBeenCalledTimes(1);

    logSpy.mockRestore();
    fs.rmSync(path.dirname(sessionPath), { recursive: true, force: true });
  });

  it('routes --active execution to active session observer', async () => {
    observeActiveSessionsMock.mockResolvedValue({
      agentId: 'clawdious',
      sessionsDir: '/tmp/sessions',
      checkedSessions: 2,
      candidateSessions: 1,
      observedSessions: 1,
      cursorUpdates: 1,
      dryRun: false,
      totalNewBytes: 2048,
      candidates: [
        {
          sessionId: 'abc',
          sessionKey: 'agent:clawdious:main',
          sourceLabel: 'main',
          filePath: '/tmp/sessions/abc.jsonl',
          fileSize: 2048,
          startOffset: 0,
          newBytes: 2048,
          thresholdBytes: 1
        }
      ]
    });
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined);

    await observeCommand({
      vaultPath: '/tmp/vault',
      active: true,
      agent: 'clawdious',
      minNew: 1,
      threshold: 10,
      reflectThreshold: 20
    });

    expect(observeActiveSessionsMock).toHaveBeenCalledWith(
      expect.objectContaining({
        vaultPath: path.resolve('/tmp/vault'),
        agentId: 'clawdious',
        minNewBytes: 1
      })
    );
    expect(processMessagesMock).not.toHaveBeenCalled();

    logSpy.mockRestore();
  });
});
