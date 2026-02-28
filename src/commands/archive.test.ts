import { afterEach, describe, expect, it, vi } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { Command } from 'commander';

const { archiveObservationsMock } = vi.hoisted(() => ({
  archiveObservationsMock: vi.fn()
}));

vi.mock('../observer/archive.js', () => ({
  archiveObservations: archiveObservationsMock
}));

import { archiveCommand, registerArchiveCommand } from './archive.js';

const createdTempDirs: string[] = [];

function makeTempVault(): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'clawvault-archive-cmd-'));
  createdTempDirs.push(dir);
  return dir;
}

afterEach(() => {
  vi.clearAllMocks();
  while (createdTempDirs.length > 0) {
    const dir = createdTempDirs.pop();
    if (dir) {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  }
});

describe('archiveCommand', () => {
  it('prints a no-op message when nothing matches archive criteria', async () => {
    archiveObservationsMock.mockReturnValue({ archived: 0, dryRun: false });
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined);

    const vaultPath = makeTempVault();
    await archiveCommand({ vaultPath, olderThan: 14 });

    expect(archiveObservationsMock).toHaveBeenCalledWith(path.resolve(vaultPath), {
      olderThanDays: 14,
      dryRun: undefined
    });
    expect(logSpy).toHaveBeenCalledWith('No observations matched archive criteria.');

    logSpy.mockRestore();
  });

  it('prints dry-run summary when archive is in dry-run mode', async () => {
    archiveObservationsMock.mockReturnValue({ archived: 3, dryRun: true });
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined);

    const vaultPath = makeTempVault();
    await archiveCommand({ vaultPath, olderThan: 30, dryRun: true });

    expect(logSpy).toHaveBeenCalledWith('Dry run: 3 observation file(s) would be archived.');
    logSpy.mockRestore();
  });

  it('prints archive summary when files were archived', async () => {
    archiveObservationsMock.mockReturnValue({ archived: 2, dryRun: false });
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined);

    const vaultPath = makeTempVault();
    await archiveCommand({ vaultPath, olderThan: 7 });

    expect(logSpy).toHaveBeenCalledWith('Archived 2 observation file(s).');
    logSpy.mockRestore();
  });
});

describe('registerArchiveCommand', () => {
  it('parses options and forwards normalized values to archiveCommand', async () => {
    archiveObservationsMock.mockReturnValue({ archived: 0, dryRun: false });
    const program = new Command();
    registerArchiveCommand(program);

    const vaultPath = makeTempVault();
    await program.parseAsync(
      ['archive', '--older-than', '21', '--dry-run', '--vault', vaultPath],
      { from: 'user' }
    );

    expect(archiveObservationsMock).toHaveBeenCalledWith(path.resolve(vaultPath), {
      olderThanDays: 21,
      dryRun: true
    });
  });

  it('throws for invalid --older-than values', async () => {
    const program = new Command();
    registerArchiveCommand(program);

    await expect(
      program.parseAsync(['archive', '--older-than', '0'], { from: 'user' })
    ).rejects.toThrow('Invalid older-than: 0');
  });
});
