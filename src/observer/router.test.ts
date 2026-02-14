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
  it('routes people observations to entity-slug subfolders', () => {
    const vaultPath = makeTempVault();
    const router = new Router(vaultPath);

    const markdown = [
      '## 2026-02-11',
      '',
      '- [relationship|c=0.80|i=0.60] 09:00 talked to Pedro about deployment cutover',
      '- [relationship|c=0.80|i=0.60] 09:10 met with Maria to review logs',
      '- [relationship|c=0.80|i=0.60] 09:20 Justin from ops mentioned latency spikes',
      '- [relationship|c=0.80|i=0.60] 09:30 Alex said rollback drills are complete'
    ].join('\n');

    try {
      const { routed } = router.route(markdown);
      const peopleItems = routed.filter((item) => item.category === 'people');
      expect(peopleItems).toHaveLength(4);

      // Each person gets their own subfolder with date-based file
      const pedroFile = path.join(vaultPath, 'people', 'pedro', '2026-02-11.md');
      expect(fs.existsSync(pedroFile)).toBe(true);
      const pedroContent = fs.readFileSync(pedroFile, 'utf-8');
      expect(pedroContent).toContain('talked to [[Pedro]]');

      const mariaFile = path.join(vaultPath, 'people', 'maria', '2026-02-11.md');
      expect(fs.existsSync(mariaFile)).toBe(true);
      expect(fs.readFileSync(mariaFile, 'utf-8')).toContain('met with [[Maria]]');

      const justinFile = path.join(vaultPath, 'people', 'justin', '2026-02-11.md');
      expect(fs.existsSync(justinFile)).toBe(true);
      expect(fs.readFileSync(justinFile, 'utf-8')).toContain('[[Justin]] from ops');

      const alexFile = path.join(vaultPath, 'people', 'alex', '2026-02-11.md');
      expect(fs.existsSync(alexFile)).toBe(true);
      expect(fs.readFileSync(alexFile, 'utf-8')).toContain('[[Alex]] said');

      // No date file at people root level
      const rootDateFile = path.join(vaultPath, 'people', '2026-02-11.md');
      expect(fs.existsSync(rootDateFile)).toBe(false);
    } finally {
      fs.rmSync(vaultPath, { recursive: true, force: true });
    }
  });

  it('does not create files at vault root', () => {
    const vaultPath = makeTempVault();
    const router = new Router(vaultPath);

    const markdown = [
      '## 2026-02-11',
      '',
      '- [decision|c=0.90|i=0.88] 09:00 decided to use PostgreSQL over SQLite',
    ].join('\n');

    try {
      router.route(markdown);
      // Should be in decisions/, not vault root
      const rootFiles = fs.readdirSync(vaultPath).filter(f => f.endsWith('.md'));
      expect(rootFiles).toHaveLength(0);
      expect(fs.existsSync(path.join(vaultPath, 'decisions', '2026-02-11.md'))).toBe(true);
    } finally {
      fs.rmSync(vaultPath, { recursive: true, force: true });
    }
  });
});
