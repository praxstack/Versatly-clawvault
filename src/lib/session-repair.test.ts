import { afterEach, describe, expect, it, vi } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import {
  analyzeSession,
  computeParentRelinks,
  extractToolUses,
  findCorruptedEntries,
  parseTranscript,
  repairSession,
  type TranscriptEntry,
} from './session-repair.js';

function createTempSession(entries: unknown[]): string {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'clawvault-session-repair-'));
  const filePath = path.join(tempDir, 'session.jsonl');
  fs.writeFileSync(filePath, entries.map((entry) => JSON.stringify(entry)).join('\n') + '\n', 'utf-8');
  return filePath;
}

function cleanup(filePath: string): void {
  fs.rmSync(path.dirname(filePath), { recursive: true, force: true });
}

function messageEntry(
  id: string,
  parentId: string | null,
  message: TranscriptEntry['message']
): TranscriptEntry {
  return {
    type: 'message',
    id,
    parentId,
    timestamp: '2026-02-14T00:00:00.000Z',
    message,
  };
}

describe('session-repair (lib)', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('parses transcript and skips malformed lines with warning', () => {
    const filePath = createTempSession([
      { type: 'session', id: 's1', parentId: null, timestamp: '2026-02-14T00:00:00.000Z' },
      { type: 'message', id: 'm1', parentId: null, timestamp: '2026-02-14T00:00:01.000Z', message: { role: 'user', content: [] } },
    ]);

    try {
      fs.appendFileSync(filePath, '{bad-json}\n');
      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

      const parsed = parseTranscript(filePath);
      expect(parsed).toHaveLength(2);
      expect(warnSpy).toHaveBeenCalledWith('Warning: Could not parse line 3');
    } finally {
      cleanup(filePath);
    }
  });

  it('extracts tool uses from multiple block types and marks partial entries as aborted', () => {
    const entries = [
      {
        line: 1,
        entry: messageEntry('m-assistant', null, {
          role: 'assistant',
          stopReason: 'stop',
          content: [
            { type: 'tool_use', id: 'tool-a', name: 'shell' },
            { type: 'functionCall', id: 'tool-b', name: 'search', partialJson: '{"q":"incomplete' },
          ],
        }),
      },
    ];

    const toolUses = extractToolUses(entries);
    expect(toolUses.get('tool-a')).toMatchObject({ isAborted: false, isPartial: false, name: 'shell' });
    expect(toolUses.get('tool-b')).toMatchObject({ isAborted: true, isPartial: true, name: 'search' });
  });

  it('detects orphaned tool results from content-level toolUseId and relinks ancestors', () => {
    const entries = [
      { line: 1, entry: { type: 'session', id: 's1', parentId: null, timestamp: '2026-02-14T00:00:00.000Z' } as TranscriptEntry },
      {
        line: 2,
        entry: messageEntry('m1', null, {
          role: 'assistant',
          stopReason: 'aborted',
          content: [{ type: 'toolCall', id: 'tool-x', name: 'exec' }],
        }),
      },
      {
        line: 3,
        entry: messageEntry('m2', 'm1', {
          role: 'toolResult',
          content: [{ type: 'text', toolUseId: 'tool-x', text: 'result' }],
        }),
      },
      {
        line: 4,
        entry: messageEntry('m3', 'm2', {
          role: 'assistant',
          content: [{ type: 'text', text: 'continue' }],
        }),
      },
    ];

    const toolUses = extractToolUses(entries);
    const { corrupted, entriesToRemove } = findCorruptedEntries(entries, toolUses);
    const relinks = computeParentRelinks(entries, entriesToRemove);

    expect(corrupted.map((entry) => entry.type)).toEqual(['aborted_tool_use', 'orphaned_tool_result']);
    expect(entriesToRemove).toEqual(new Set(['m1', 'm2']));
    expect(relinks).toEqual([
      {
        lineNumber: 4,
        entryId: 'm3',
        oldParentId: 'm2',
        newParentId: 'null',
      },
    ]);
  });

  it('analyzes and repairs corrupted transcripts with dry-run and write modes', () => {
    const filePath = createTempSession([
      { type: 'session', id: 'sess-1', parentId: null, timestamp: '2026-02-14T00:00:00.000Z' },
      messageEntry('m1', null, { role: 'user', content: [{ type: 'text', text: 'run tool' }] }),
      messageEntry('m2', 'm1', {
        role: 'assistant',
        stopReason: 'aborted',
        content: [{ type: 'toolCall', id: 'tool-1', name: 'exec', partialJson: '{"cmd":"ls' }],
      }),
      messageEntry('m3', 'm2', {
        role: 'toolResult',
        content: [{ type: 'text', toolCallId: 'tool-1', text: 'partial' }],
      }),
      messageEntry('m4', 'm3', { role: 'assistant', content: [{ type: 'text', text: 'fallback' }] }),
    ]);

    try {
      const original = fs.readFileSync(filePath, 'utf-8');
      const analysis = analyzeSession(filePath);
      expect(analysis.sessionId).toBe('sess-1');
      expect(analysis.removedCount).toBe(2);
      expect(analysis.relinkedCount).toBe(1);

      const dryRun = repairSession(filePath, { dryRun: true, backup: false });
      expect(dryRun.repaired).toBe(false);
      expect(fs.readFileSync(filePath, 'utf-8')).toBe(original);

      const repaired = repairSession(filePath, { backup: false });
      expect(repaired.repaired).toBe(true);
      expect(repaired.removedCount).toBe(2);
      expect(repaired.relinkedCount).toBe(1);

      const parsed = parseTranscript(filePath);
      expect(parsed.map((entry) => entry.entry.id)).toEqual(['sess-1', 'm1', 'm4']);
      expect(parsed.find((entry) => entry.entry.id === 'm4')?.entry.parentId).toBe('m1');
      expect(fs.readFileSync(filePath, 'utf-8').endsWith('\n')).toBe(true);
    } finally {
      cleanup(filePath);
    }
  });
});
