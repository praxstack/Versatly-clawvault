import { afterEach, describe, expect, it, vi } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

const { spawnSyncMock } = vi.hoisted(() => ({
  spawnSyncMock: vi.fn()
}));

vi.mock('child_process', () => ({
  spawnSync: spawnSyncMock
}));

async function loadCompatModule() {
  vi.resetModules();
  return await import('./compat.js');
}

function writeProjectFixture(root: string): void {
  fs.mkdirSync(path.join(root, 'hooks', 'clawvault'), { recursive: true });
  fs.writeFileSync(
    path.join(root, 'package.json'),
    JSON.stringify({
      name: 'clawvault',
      openclaw: { hooks: ['./hooks/clawvault'] }
    }),
    'utf-8'
  );
  fs.writeFileSync(
    path.join(root, 'SKILL.md'),
    '---\nmetadata: {"openclaw":{"emoji":"🐘"}}\n---',
    'utf-8'
  );
  fs.writeFileSync(
    path.join(root, 'hooks', 'clawvault', 'HOOK.md'),
    [
      '---',
      'metadata:',
      '  openclaw:',
      '    events: ["gateway:startup","command:new","session:start"]',
      '---'
    ].join('\n'),
    'utf-8'
  );
  fs.writeFileSync(
    path.join(root, 'hooks', 'clawvault', 'handler.js'),
    "import { execFileSync } from 'child_process';\nexecFileSync('clawvault', ['wake']);\n",
    'utf-8'
  );
}

afterEach(() => {
  vi.clearAllMocks();
});

describe('checkOpenClawCompatibility', () => {
  it('returns healthy report for valid fixtures', async () => {
    spawnSyncMock.mockReturnValue({ error: undefined });
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'clawvault-compat-'));
    try {
      writeProjectFixture(root);
      const { checkOpenClawCompatibility } = await loadCompatModule();
      const report = checkOpenClawCompatibility({ baseDir: root });
      expect(report.errors).toBe(0);
      expect(report.warnings).toBe(0);
    } finally {
      fs.rmSync(root, { recursive: true, force: true });
    }
  });

  it('flags missing openclaw binary as warning', async () => {
    spawnSyncMock.mockReturnValue({ error: new Error('missing') });
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'clawvault-compat-'));
    try {
      writeProjectFixture(root);
      const { checkOpenClawCompatibility } = await loadCompatModule();
      const report = checkOpenClawCompatibility({ baseDir: root });
      const cliCheck = report.checks.find((check) => check.label === 'openclaw CLI available');
      expect(cliCheck?.status).toBe('warn');
      expect(report.warnings).toBeGreaterThanOrEqual(1);
    } finally {
      fs.rmSync(root, { recursive: true, force: true });
    }
  });
});
