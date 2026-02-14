import { describe, expect, it } from 'vitest';
import { Command } from 'commander';
import { registerTaskCommands } from './register-task-commands.js';
import { chalkStub, stubResolveVaultPath } from './test-helpers/cli-command-fixtures.js';

describe('register-task-commands', () => {
  it('adds canvas template and listing flags', () => {
    const program = new Command();
    registerTaskCommands(program, {
      chalk: chalkStub,
      resolveVaultPath: stubResolveVaultPath
    });

    const canvasCommand = program.commands.find((command) => command.name() === 'canvas');
    expect(canvasCommand).toBeDefined();

    const optionFlags = canvasCommand?.options.map((option) => option.flags) ?? [];
    expect(optionFlags).toEqual(expect.arrayContaining([
      '--template <id>',
      '--list-templates',
      '--project <project>'
    ]));
  });
});
