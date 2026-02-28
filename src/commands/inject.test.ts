import { afterEach, describe, expect, it, vi } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { Command } from 'commander';
import type { InjectResult } from '../lib/inject-utils.js';

const { listConfigMock, runPromptInjectionMock } = vi.hoisted(() => ({
  listConfigMock: vi.fn(),
  runPromptInjectionMock: vi.fn()
}));

vi.mock('../lib/config-manager.js', () => ({
  listConfig: listConfigMock
}));

vi.mock('../lib/inject-utils.js', () => ({
  runPromptInjection: runPromptInjectionMock
}));

import { buildInjectionResult, injectCommand, registerInjectCommand } from './inject.js';

const createdTempDirs: string[] = [];

function makeTempVault(): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'clawvault-inject-cmd-'));
  createdTempDirs.push(dir);
  return dir;
}

function makeInjectResult(overrides: Partial<InjectResult> = {}): InjectResult {
  return {
    message: 'default message',
    generatedAt: '2026-02-20T00:00:00.000Z',
    deterministicMs: 12,
    llmProvider: null,
    usedLlm: false,
    matches: [],
    ...overrides
  };
}

afterEach(() => {
  vi.clearAllMocks();
  vi.restoreAllMocks();
  while (createdTempDirs.length > 0) {
    const dir = createdTempDirs.pop();
    if (dir) {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  }
});

describe('buildInjectionResult', () => {
  it('throws when message is blank after trimming', async () => {
    const vaultPath = makeTempVault();
    await expect(
      buildInjectionResult('   ', { vaultPath })
    ).rejects.toThrow('Message is required for inject.');
  });

  it('uses injected config defaults when explicit options are missing', async () => {
    const vaultPath = makeTempVault();
    listConfigMock.mockReturnValue({
      inject: {
        maxResults: '5',
        useLlm: 'false',
        scope: 'global,team'
      }
    });
    runPromptInjectionMock.mockResolvedValue(makeInjectResult());

    await buildInjectionResult('   ship the release   ', { vaultPath });

    expect(runPromptInjectionMock).toHaveBeenCalledWith(
      path.resolve(vaultPath),
      'ship the release',
      {
        maxResults: 5,
        useLlm: false,
        scope: ['global', 'team'],
        model: undefined
      }
    );
  });

  it('prefers explicit options over config defaults', async () => {
    const vaultPath = makeTempVault();
    listConfigMock.mockReturnValue({
      inject: {
        maxResults: 2,
        useLlm: true,
        scope: ['global']
      }
    });
    runPromptInjectionMock.mockResolvedValue(makeInjectResult());

    await buildInjectionResult('deploy patch', {
      vaultPath,
      maxResults: 9,
      useLlm: false,
      scope: 'decisions',
      model: 'gpt-5-mini'
    });

    expect(runPromptInjectionMock).toHaveBeenCalledWith(
      path.resolve(vaultPath),
      'deploy patch',
      {
        maxResults: 9,
        useLlm: false,
        scope: 'decisions',
        model: 'gpt-5-mini'
      }
    );
  });
});

describe('injectCommand', () => {
  it('prints markdown output by default', async () => {
    const vaultPath = makeTempVault();
    listConfigMock.mockReturnValue({ inject: { maxResults: 8, useLlm: true, scope: ['global'] } });
    runPromptInjectionMock.mockResolvedValue(makeInjectResult({
      message: 'How should we deploy this?',
      deterministicMs: 27,
      usedLlm: true,
      llmProvider: 'openai',
      matches: [{
        item: {
          id: 'rules/deploy.md',
          category: 'rules',
          relativePath: 'rules/deploy.md',
          title: 'Deploy Safely',
          content: 'Always roll out gradually.',
          triggers: ['deploy'],
          scope: ['global'],
          priority: 100,
          searchKeywords: ['deploy'],
          noteNodeId: 'note:rules/deploy'
        },
        score: 142.35,
        deterministicScore: 118,
        llmScore: 0.9,
        reasons: [
          { source: 'trigger', value: 'deploy', weight: 18 },
          { source: 'llm_intent', value: 'relevant to deployment', weight: 24 }
        ]
      }]
    }));
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined);

    await injectCommand('How should we deploy this?', {
      vaultPath
    });

    const output = String(logSpy.mock.calls[0][0]);
    expect(output).toContain('## Prompt Injection for: How should we deploy this?');
    expect(output).toContain('Deterministic matching: 27ms');
    expect(output).toContain('LLM fuzzy matching: enabled (openai)');
    expect(output).toContain('### 1. Deploy Safely');
    expect(output).toContain('Match sources: trigger:deploy | llm_intent:relevant to deployment');
    expect(output).toContain('Always roll out gradually.');
  });

  it('registerInjectCommand parses flags and supports JSON output', async () => {
    const vaultPath = makeTempVault();
    listConfigMock.mockReturnValue({ inject: { maxResults: 8, useLlm: true, scope: ['global'] } });
    runPromptInjectionMock.mockResolvedValue(makeInjectResult({
      message: 'Plan migration',
      matches: [{
        item: {
          id: 'decisions/migrate.md',
          category: 'decisions',
          relativePath: 'decisions/migrate.md',
          title: 'Migration Policy',
          content: 'Use phased rollout.',
          triggers: ['migration'],
          scope: ['global'],
          priority: 80,
          searchKeywords: ['migration'],
          noteNodeId: 'note:decisions/migrate'
        },
        score: 88,
        deterministicScore: 88,
        llmScore: null,
        reasons: [{ source: 'keyword', value: 'migration', weight: 8 }]
      }]
    }));

    const program = new Command();
    registerInjectCommand(program);
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined);

    await program.parseAsync([
      'inject',
      'Plan migration',
      '--vault',
      vaultPath,
      '--max-results',
      '3',
      '--scope',
      'rules,team',
      '--disable-llm',
      '--format',
      'json',
      '--model',
      'gpt-5-mini'
    ], { from: 'user' });

    expect(runPromptInjectionMock).toHaveBeenCalledWith(
      path.resolve(vaultPath),
      'Plan migration',
      {
        maxResults: 3,
        useLlm: false,
        scope: 'rules,team',
        model: 'gpt-5-mini'
      }
    );

    const output = JSON.parse(String(logSpy.mock.calls[0][0])) as InjectResult;
    expect(output.message).toBe('Plan migration');
    expect(output.matches).toHaveLength(1);
  });
});
