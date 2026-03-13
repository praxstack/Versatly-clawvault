import { describe, expect, it, vi } from 'vitest';
import { Command } from 'commander';
import * as fs from 'fs';
import * as path from 'path';
import { registerCoreCommands } from './register-core-commands.js';
import { chalkStub } from './test-helpers/cli-command-fixtures.js';

function buildProgram(patchImpl) {
  const program = new Command();
  registerCoreCommands(program, {
    chalk: chalkStub,
    path,
    fs,
    createVault: async () => ({ getCategories: () => [], getQmdRoot: () => '', getQmdCollection: () => '' }),
    getVault: async () => ({ patch: patchImpl }),
    runQmd: async () => {}
  });
  return program;
}

describe('register-core-commands patch command', () => {
  it('exposes patch command mode flags', () => {
    const program = buildProgram(async () => ({ id: 'decisions/example', path: '/vault/decisions/example.md' }));
    const patchCommand = program.commands.find((command) => command.name() === 'patch');
    expect(patchCommand).toBeDefined();
    const optionFlags = patchCommand?.options.map((option) => option.flags) ?? [];
    expect(optionFlags).toEqual(expect.arrayContaining([
      '--append <text>',
      '--replace <text>',
      '--with <text>',
      '--section <heading>',
      '--content <text>',
      '-v, --vault <path>'
    ]));
  });

  it('forwards append mode payload to vault.patch', async () => {
    const patchMock = vi.fn(async () => ({ id: 'decisions/example', path: '/vault/decisions/example.md' }));
    const program = buildProgram(patchMock);
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    try {
      await program.parseAsync(['patch', 'decisions/example', '--append', 'new line'], { from: 'user' });
      expect(patchMock).toHaveBeenCalledWith({
        idOrPath: 'decisions/example',
        mode: 'append',
        append: 'new line'
      });
      expect(errorSpy).not.toHaveBeenCalled();
    } finally {
      logSpy.mockRestore();
      errorSpy.mockRestore();
    }
  });

  it('forwards section/content mode payload to vault.patch', async () => {
    const patchMock = vi.fn(async () => ({ id: 'decisions/example', path: '/vault/decisions/example.md' }));
    const program = buildProgram(patchMock);
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    try {
      await program.parseAsync(
        ['patch', 'decisions/example', '--section', 'Notes', '--content', 'updated notes'],
        { from: 'user' }
      );
      expect(patchMock).toHaveBeenCalledWith({
        idOrPath: 'decisions/example',
        mode: 'content',
        content: 'updated notes',
        section: 'Notes'
      });
      expect(errorSpy).not.toHaveBeenCalled();
    } finally {
      logSpy.mockRestore();
      errorSpy.mockRestore();
    }
  });
});
