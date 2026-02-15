/**
 * Project command registrations for ClawVault
 * Registers project add/update/archive/list/show/tasks/board commands
 */

function parseCsvList(value) {
  if (!value) return undefined;
  const items = String(value)
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean);
  return items.length > 0 ? items : undefined;
}

export function registerProjectCommands(
  program,
  { chalk, resolveVaultPath }
) {
  const projectCmd = program
    .command('project')
    .description('Project management');

  projectCmd
    .command('add <title>')
    .description('Add a new project')
    .option('-v, --vault <path>', 'Vault path')
    .option('--owner <owner>', 'Project owner')
    .option('--status <status>', 'Project status (active, paused, completed, archived)')
    .option('--team <team>', 'Comma-separated team members')
    .option('--client <client>', 'Client name')
    .option('--tags <tags>', 'Comma-separated tags')
    .option('--description <description>', 'One-line project summary')
    .option('--deadline <date>', 'Deadline (YYYY-MM-DD)')
    .option('--repo <url>', 'Repository URL')
    .option('--url <url>', 'Production URL')
    .action(async (title, options) => {
      try {
        const vaultPath = resolveVaultPath(options.vault);
        const { projectCommand } = await import('../dist/commands/project.js');
        await projectCommand(vaultPath, 'add', {
          title,
          options: {
            owner: options.owner,
            status: options.status,
            team: parseCsvList(options.team),
            client: options.client,
            tags: parseCsvList(options.tags),
            description: options.description,
            deadline: options.deadline,
            repo: options.repo,
            url: options.url
          }
        });
      } catch (err) {
        console.error(chalk.red(`Error: ${err.message}`));
        process.exit(1);
      }
    });

  projectCmd
    .command('update <slug>')
    .description('Update a project')
    .option('-v, --vault <path>', 'Vault path')
    .option('--status <status>', 'Project status (active, paused, completed, archived)')
    .option('--owner <owner>', 'Project owner')
    .option('--team <team>', 'Comma-separated team members')
    .option('--client <client>', 'Client name')
    .option('--tags <tags>', 'Comma-separated tags')
    .option('--description <description>', 'One-line project summary')
    .option('--deadline <date>', 'Deadline (YYYY-MM-DD)')
    .option('--repo <url>', 'Repository URL')
    .option('--url <url>', 'Production URL')
    .action(async (slug, options) => {
      try {
        const vaultPath = resolveVaultPath(options.vault);
        const { projectCommand } = await import('../dist/commands/project.js');
        await projectCommand(vaultPath, 'update', {
          slug,
          options: {
            status: options.status,
            owner: options.owner,
            team: parseCsvList(options.team),
            client: options.client,
            tags: parseCsvList(options.tags),
            description: options.description,
            deadline: options.deadline,
            repo: options.repo,
            url: options.url
          }
        });
      } catch (err) {
        console.error(chalk.red(`Error: ${err.message}`));
        process.exit(1);
      }
    });

  projectCmd
    .command('archive <slug>')
    .description('Archive a project')
    .option('-v, --vault <path>', 'Vault path')
    .option('--reason <reason>', 'Reason for archiving')
    .action(async (slug, options) => {
      try {
        const vaultPath = resolveVaultPath(options.vault);
        const { projectCommand } = await import('../dist/commands/project.js');
        await projectCommand(vaultPath, 'archive', {
          slug,
          options: {
            reason: options.reason
          }
        });
      } catch (err) {
        console.error(chalk.red(`Error: ${err.message}`));
        process.exit(1);
      }
    });

  projectCmd
    .command('list')
    .description('List projects')
    .option('-v, --vault <path>', 'Vault path')
    .option('--status <status>', 'Filter by status')
    .option('--owner <owner>', 'Filter by owner')
    .option('--client <client>', 'Filter by client')
    .option('--tag <tag>', 'Filter by tag')
    .option('--json', 'Output as JSON')
    .action(async (options) => {
      try {
        const vaultPath = resolveVaultPath(options.vault);
        const { projectCommand } = await import('../dist/commands/project.js');
        await projectCommand(vaultPath, 'list', {
          options: {
            status: options.status,
            owner: options.owner,
            client: options.client,
            tag: options.tag,
            json: options.json
          }
        });
      } catch (err) {
        console.error(chalk.red(`Error: ${err.message}`));
        process.exit(1);
      }
    });

  projectCmd
    .command('show <slug>')
    .description('Show project details')
    .option('-v, --vault <path>', 'Vault path')
    .option('--json', 'Output as JSON')
    .action(async (slug, options) => {
      try {
        const vaultPath = resolveVaultPath(options.vault);
        const { projectCommand } = await import('../dist/commands/project.js');
        await projectCommand(vaultPath, 'show', {
          slug,
          options: {
            json: options.json
          }
        });
      } catch (err) {
        console.error(chalk.red(`Error: ${err.message}`));
        process.exit(1);
      }
    });

  projectCmd
    .command('tasks <slug>')
    .description('List tasks for a project')
    .option('-v, --vault <path>', 'Vault path')
    .option('--json', 'Output as JSON')
    .action(async (slug, options) => {
      try {
        const vaultPath = resolveVaultPath(options.vault);
        const { projectCommand } = await import('../dist/commands/project.js');
        await projectCommand(vaultPath, 'tasks', {
          slug,
          options: {
            json: options.json
          }
        });
      } catch (err) {
        console.error(chalk.red(`Error: ${err.message}`));
        process.exit(1);
      }
    });

  projectCmd
    .command('board')
    .description('Generate project kanban board')
    .option('-v, --vault <path>', 'Vault path')
    .option('--output <path>', 'Board markdown path (default: Projects-Board.md)')
    .option('--group-by <field>', 'Grouping field (status, owner, client)')
    .action(async (options) => {
      try {
        const vaultPath = resolveVaultPath(options.vault);
        const { projectCommand } = await import('../dist/commands/project.js');
        await projectCommand(vaultPath, 'board', {
          options: {
            output: options.output,
            groupBy: options.groupBy
          }
        });
      } catch (err) {
        console.error(chalk.red(`Error: ${err.message}`));
        process.exit(1);
      }
    });
}
