import { describe, expect, it } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { Router } from './router.js';

function makeTempVault(): string {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'clawvault-router-'));
  fs.writeFileSync(path.join(root, '.clawvault.json'), JSON.stringify({ name: 'test' }));
  return root;
}

describe('Router', () => {
  it('routes people observations from name and phrase patterns', () => {
    const vaultPath = makeTempVault();
    const router = new Router(vaultPath);

    const markdown = [
      '## 2026-02-11',
      '',
      '🟡 09:00 talked to Pedro about deployment cutover',
      '🟡 09:10 met with Maria to review logs',
      '🟡 09:20 Justin from ops mentioned latency spikes',
      '🟡 09:30 Alex said rollback drills are complete'
    ].join('\n');

    try {
      const { routed } = router.route(markdown);
      const peopleItems = routed.filter((item) => item.category === 'people');
      expect(peopleItems).toHaveLength(4);

      const peopleFile = path.join(vaultPath, 'people', '2026-02-11.md');
      expect(fs.existsSync(peopleFile)).toBe(true);
      const content = fs.readFileSync(peopleFile, 'utf-8');
      expect(content).toContain('talked to [[Pedro]]');
      expect(content).toContain('met with [[Maria]]');
      expect(content).toContain('[[Justin]] from ops mentioned latency spikes');
      expect(content).toContain('[[Alex]] said rollback drills are complete');
    } finally {
      fs.rmSync(vaultPath, { recursive: true, force: true });
    }
  });
});
