import { afterEach, describe, expect, it, vi } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { Command } from 'commander';

const { runReflectionMock } = vi.hoisted(() => ({
  runReflectionMock: vi.fn()
}));

vi.mock('../observer/reflection-service.js', () => ({
  runReflection: runReflectionMock
}));

import { reflectCommand, registerReflectCommand } from './reflect.js';

const createdTempDirs: string[] = [];

function makeTempVault(): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'clawvault-reflect-cmd-'));
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

describe('reflectCommand', () => {
  it('prints a no-op message when no reflections are promoted', async () => {
    runReflectionMock.mockResolvedValue({
      processedWeeks: 1,
      writtenWeeks: 0,
      dryRun: false,
      files: [],
      archive: null
    });
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined);

    const vaultPath = makeTempVault();
    await reflectCommand({ vaultPath, days: 14 });

    expect(runReflectionMock).toHaveBeenCalledWith({
      vaultPath: path.resolve(vaultPath),
      days: 14,
      dryRun: undefined
    });
    expect(logSpy).toHaveBeenCalledWith('No new reflections promoted.');
    logSpy.mockRestore();
  });

  it('prints dry-run summary when reflection is in dry-run mode', async () => {
    runReflectionMock.mockResolvedValue({
      processedWeeks: 2,
      writtenWeeks: 3,
      dryRun: true,
      files: ['/tmp/reflections/2026-W08.md'],
      archive: null
    });
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined);

    const vaultPath = makeTempVault();
    await reflectCommand({ vaultPath, days: 21, dryRun: true });

    expect(logSpy).toHaveBeenCalledWith('Dry run: 3 reflection file(s) would be written.');
    logSpy.mockRestore();
  });

  it('prints completion and archive summaries when reflections are written', async () => {
    runReflectionMock.mockResolvedValue({
      processedWeeks: 2,
      writtenWeeks: 1,
      dryRun: false,
      files: ['/tmp/reflections/2026-W08.md'],
      archive: {
        archived: 4,
        dryRun: false
      }
    });
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined);

    const vaultPath = makeTempVault();
    await reflectCommand({ vaultPath, days: 7 });

    expect(logSpy).toHaveBeenCalledWith('Reflection complete: 1 week file(s) updated.');
    expect(logSpy).toHaveBeenCalledWith('Archive pass: 4 observation file(s) archived.');
    logSpy.mockRestore();
  });
});

describe('registerReflectCommand', () => {
  it('parses options and forwards normalized values to reflectCommand', async () => {
    runReflectionMock.mockResolvedValue({
      processedWeeks: 0,
      writtenWeeks: 0,
      dryRun: true,
      files: [],
      archive: null
    });
    const program = new Command();
    registerReflectCommand(program);

    const vaultPath = makeTempVault();
    await program.parseAsync(
      ['reflect', '--days', '30', '--dry-run', '--vault', vaultPath],
      { from: 'user' }
    );

    expect(runReflectionMock).toHaveBeenCalledWith({
      vaultPath: path.resolve(vaultPath),
      days: 30,
      dryRun: true
    });
  });

  it('throws for invalid --days values', async () => {
    const program = new Command();
    registerReflectCommand(program);

    await expect(
      program.parseAsync(['reflect', '--days', '-1'], { from: 'user' })
    ).rejects.toThrow('Invalid days: -1');
  });
});
