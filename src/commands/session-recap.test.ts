import { afterEach, describe, expect, it, vi } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { buildSessionRecap, sessionRecapCommand } from './session-recap.js';

const originalOpenClawStateDir = process.env.OPENCLAW_STATE_DIR;
const originalOpenClawAgentId = process.env.OPENCLAW_AGENT_ID;
const createdTempDirs: string[] = [];

interface SessionFixtureOptions {
  agentId?: string;
  sessionKey?: string;
  sessionId?: string;
  sessionFile?: string;
  transcriptLines?: string[];
  includeCanonicalTranscript?: boolean;
}

interface SessionFixture {
  root: string;
  agentId: string;
  sessionKey: string;
  sessionId: string;
  sessionsDir: string;
  transcriptPath: string;
}

function makeTempDir(prefix: string): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), prefix));
  createdTempDirs.push(dir);
  return dir;
}

function messageLine(role: string, content: unknown): string {
  return JSON.stringify({
    type: 'message',
    message: {
      role,
      content
    }
  });
}

function setupSessionFixture(options: SessionFixtureOptions = {}): SessionFixture {
  const root = makeTempDir('clawvault-session-recap-');
  process.env.OPENCLAW_STATE_DIR = root;

  const agentId = options.agentId ?? 'clawdious';
  const sessionKey = options.sessionKey ?? `agent:${agentId}:main`;
  const sessionId = options.sessionId ?? 'session-001';

  const sessionsDir = path.join(root, 'agents', agentId, 'sessions');
  fs.mkdirSync(sessionsDir, { recursive: true });
  const transcriptPath = path.join(sessionsDir, `${sessionId}.jsonl`);

  fs.writeFileSync(
    path.join(sessionsDir, 'sessions.json'),
    JSON.stringify({
      [sessionKey]: {
        sessionId,
        sessionFile: options.sessionFile
      }
    }),
    'utf-8'
  );

  if (options.includeCanonicalTranscript !== false) {
    const lines = options.transcriptLines ?? [];
    fs.writeFileSync(transcriptPath, `${lines.join('\n')}\n`, 'utf-8');
  }

  return {
    root,
    agentId,
    sessionKey,
    sessionId,
    sessionsDir,
    transcriptPath
  };
}

afterEach(() => {
  vi.restoreAllMocks();
  process.env.OPENCLAW_STATE_DIR = originalOpenClawStateDir;
  process.env.OPENCLAW_AGENT_ID = originalOpenClawAgentId;
  while (createdTempDirs.length > 0) {
    const dir = createdTempDirs.pop();
    if (dir) {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  }
});

describe('buildSessionRecap', () => {
  it('extracts recent user and assistant messages from transcript', async () => {
    const fixture = setupSessionFixture({
      transcriptLines: [
        'this is not json',
        messageLine('user', '  Need help with rollout  '),
        messageLine('assistant', [
          { type: 'thinking', text: 'internal' },
          { type: 'text', text: '  Start with canary deploys.  ' },
          { type: 'tool_result', text: 'skip tool output' }
        ]),
        messageLine('assistant', [{ type: 'tool_use', text: 'only tool metadata' }]),
        messageLine('system', 'ignore this'),
        messageLine('user', 'Can we add rollback steps?')
      ]
    });

    const result = await buildSessionRecap(fixture.sessionKey, { limit: 10 });

    expect(result.sessionLabel).toBe('main');
    expect(result.count).toBe(3);
    expect(result.messages).toEqual([
      { role: 'user', text: 'Need help with rollout' },
      { role: 'assistant', text: 'Start with canary deploys.' },
      { role: 'user', text: 'Can we add rollback steps?' }
    ]);
    expect(result.markdown).toContain('## Session Recap: main');
    expect(result.markdown).toContain('### Recent Conversation (last 3 messages)');
  });

  it('outputs JSON payload when sessionRecapCommand is called with json format', async () => {
    const fixture = setupSessionFixture({
      transcriptLines: [
        messageLine('user', 'Summarize this session'),
        messageLine('assistant', 'Here is a summary.')
      ]
    });
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined);

    await sessionRecapCommand(fixture.sessionKey, { format: 'json' });

    const payload = JSON.parse(String(logSpy.mock.calls[0][0])) as {
      sessionKey: string;
      count: number;
      messages: Array<{ role: string; text: string }>;
      markdown?: string;
    };
    expect(payload.sessionKey).toBe(fixture.sessionKey);
    expect(payload.count).toBe(2);
    expect(payload.messages).toHaveLength(2);
    expect(payload).not.toHaveProperty('markdown');
  });

  it('rejects invalid session keys', async () => {
    await expect(buildSessionRecap('invalid-session-key')).rejects.toThrow(
      'Invalid session key. Expected format: agent:<agentId>:<scope>'
    );
  });

  it('uses canonical transcript path when mapped sessionFile points outside sessions dir', async () => {
    const outsideRoot = makeTempDir('clawvault-session-recap-outside-');
    const outsideTranscript = path.join(outsideRoot, 'outside.jsonl');
    fs.writeFileSync(outsideTranscript, `${messageLine('user', 'outside transcript')}\n`, 'utf-8');

    const fixture = setupSessionFixture({
      agentId: 'safe-agent',
      sessionKey: 'agent:other-agent:incident',
      sessionId: 'safe-session',
      sessionFile: outsideTranscript,
      transcriptLines: [messageLine('user', 'inside transcript')]
    });

    const result = await buildSessionRecap(fixture.sessionKey, { agentId: 'safe-agent' });

    expect(result.agentId).toBe('safe-agent');
    expect(result.transcriptPath).toBe(path.resolve(fixture.transcriptPath));
    expect(result.messages).toEqual([{ role: 'user', text: 'inside transcript' }]);
  });

  it('caps requested limit to 50 recent messages', async () => {
    const transcriptLines: string[] = [];
    for (let index = 1; index <= 80; index += 1) {
      const role = index % 2 === 0 ? 'assistant' : 'user';
      transcriptLines.push(messageLine(role, `msg-${index}`));
    }
    const fixture = setupSessionFixture({ transcriptLines });

    const result = await buildSessionRecap(fixture.sessionKey, { limit: 999 });

    expect(result.count).toBe(50);
    expect(result.messages).toHaveLength(50);
    expect(result.messages[0]?.text).toBe('msg-31');
    expect(result.messages[result.messages.length - 1]?.text).toBe('msg-80');
  });
});
