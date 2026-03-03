/**
 * Workgraph CLI commands — multi-agent coordination primitives.
 *
 * Commands under 'wg' namespace (beautiful, agent-native):
 *   wg status                Agent morning briefing
 *   wg thread create         Create a new thread
 *   wg thread list           List threads with filters
 *   wg thread claim          Claim a thread
 *   wg thread done           Mark thread complete
 *   wg thread block          Block thread on dependency
 *   wg thread release        Release thread back to pool
 *   wg thread decompose      Break thread into sub-threads
 *   wg ledger                View coordination history
 *   wg define                Define new primitive type
 *   wg types                 List all primitive types
 *   wg create                Create any primitive
 *   wg board                 Terminal kanban board
 *
 * Legacy commands (still available):
 *   thread create <title>    Create a new thread
 *   thread list              List threads (filterable by status)
 *   thread show <path>       Show thread details + ledger history
 *   thread claim <path>      Claim a thread for this agent
 *   thread release <path>    Release a claimed thread
 *   thread done <path>       Mark thread complete
 *   thread block <path>      Mark thread blocked
 *   thread unblock <path>    Unblock a thread
 *   thread decompose <path>  Break into sub-threads
 *   primitive define <name>  Define a new primitive type
 *   primitive list           List all primitive types
 *   primitive create <type>  Create an instance of any type
 *   ledger show              Show recent ledger entries
 *   ledger history <path>    Show history of a specific file
 */

import * as os from 'os';
import * as path from 'path';

export function registerWorkgraphCommands(program, { chalk, resolveVaultPath }) {
  // Register the new 'wg' namespace with beautiful, agent-native commands
  registerWgCommands(program, { chalk, resolveVaultPath });

  // Keep legacy commands for backward compatibility
  const agentName = process.env.CLAWVAULT_AGENT || process.env.USER || 'anonymous';

  // =========================================================================
  // thread
  // =========================================================================
  const threadCmd = program
    .command('thread')
    .description('Coordinate work through threads (workgraph core)');

  threadCmd
    .command('create <title>')
    .description('Create a new thread')
    .requiredOption('-g, --goal <goal>', 'What success looks like')
    .option('-v, --vault <path>', 'Vault path')
    .option('-a, --actor <name>', 'Agent name', agentName)
    .option('-p, --priority <level>', 'urgent | high | medium | low', 'medium')
    .option('--deps <paths>', 'Comma-separated dependency thread paths')
    .option('--parent <path>', 'Parent thread path')
    .option('--context <refs>', 'Comma-separated vault doc refs for context')
    .option('--tags <tags>', 'Comma-separated tags')
    .action(async (title, opts) => {
      try {
        const vaultPath = resolveVaultPath(opts.vault);
        const { thread } = await import('../dist/workgraph/index.js');
        const t = thread.createThread(vaultPath, title, opts.goal, opts.actor, {
          priority: opts.priority,
          deps: csv(opts.deps),
          parent: opts.parent,
          context_refs: csv(opts.context),
          tags: csv(opts.tags),
        });
        console.log(chalk.green(`✓ Thread created: ${t.path}`));
        console.log(`  Title:    ${t.fields.title}`);
        console.log(`  Status:   ${t.fields.status}`);
        console.log(`  Priority: ${t.fields.priority}`);
      } catch (err) {
        console.error(chalk.red(`Error: ${err.message}`));
        process.exit(1);
      }
    });

  threadCmd
    .command('list')
    .description('List threads')
    .option('-v, --vault <path>', 'Vault path')
    .option('-s, --status <status>', 'Filter: open | active | blocked | done | cancelled')
    .option('--json', 'Output as JSON')
    .action(async (opts) => {
      try {
        const vaultPath = resolveVaultPath(opts.vault);
        const { store } = await import('../dist/workgraph/index.js');
        let threads = store.list(vaultPath, 'thread');
        if (opts.status) threads = threads.filter(t => t.fields.status === opts.status);

        if (opts.json) {
          console.log(JSON.stringify(threads.map(t => ({ path: t.path, ...t.fields })), null, 2));
          return;
        }

        if (threads.length === 0) {
          console.log(chalk.dim('No threads found.'));
          return;
        }

        const statusColor = { open: 'blue', active: 'yellow', blocked: 'red', done: 'green', cancelled: 'dim' };
        for (const t of threads) {
          const s = String(t.fields.status);
          const colorFn = chalk[statusColor[s]] || chalk.white;
          const owner = t.fields.owner ? chalk.dim(` (${t.fields.owner})`) : '';
          console.log(`  ${colorFn(`[${s}]`)} ${t.fields.title}${owner}`);
          console.log(chalk.dim(`         ${t.path}`));
        }
        console.log(chalk.dim(`\n${threads.length} thread(s)`));
      } catch (err) {
        console.error(chalk.red(`Error: ${err.message}`));
        process.exit(1);
      }
    });

  threadCmd
    .command('show <path>')
    .description('Show thread details and ledger history')
    .option('-v, --vault <path>', 'Vault path')
    .action(async (threadPath, opts) => {
      try {
        const vaultPath = resolveVaultPath(opts.vault);
        const { store, ledger } = await import('../dist/workgraph/index.js');
        const t = store.read(vaultPath, threadPath);
        if (!t) { console.error(chalk.red(`Not found: ${threadPath}`)); process.exit(1); }

        console.log(chalk.bold(String(t.fields.title)));
        console.log(chalk.dim('─'.repeat(50)));
        console.log(`Status:   ${t.fields.status}`);
        console.log(`Owner:    ${t.fields.owner || chalk.dim('unclaimed')}`);
        console.log(`Priority: ${t.fields.priority}`);
        if (t.fields.deps?.length) console.log(`Deps:     ${(t.fields.deps).join(', ')}`);
        if (t.fields.parent) console.log(`Parent:   ${t.fields.parent}`);
        if (t.fields.tags?.length) console.log(`Tags:     ${(t.fields.tags).join(', ')}`);
        console.log(`Path:     ${t.path}`);
        console.log();
        if (t.body) console.log(t.body);

        const history = ledger.historyOf(vaultPath, threadPath);
        if (history.length > 0) {
          console.log(chalk.dim('\n─── Ledger History ───'));
          for (const e of history) {
            const time = new Date(e.ts).toLocaleTimeString();
            const data = e.data ? chalk.dim(` ${JSON.stringify(e.data)}`) : '';
            console.log(`  ${chalk.dim(time)} ${chalk.cyan(e.op)} by ${e.actor}${data}`);
          }
        }
      } catch (err) {
        console.error(chalk.red(`Error: ${err.message}`));
        process.exit(1);
      }
    });

  threadCmd
    .command('claim <path>')
    .description('Claim a thread — only you can work on it')
    .option('-v, --vault <path>', 'Vault path')
    .option('-a, --actor <name>', 'Agent name', agentName)
    .action(async (threadPath, opts) => {
      try {
        const vaultPath = resolveVaultPath(opts.vault);
        const { thread } = await import('../dist/workgraph/index.js');
        const t = thread.claim(vaultPath, threadPath, opts.actor);
        console.log(chalk.green(`✓ Claimed: ${threadPath}`));
        console.log(`  Owner: ${opts.actor}`);
      } catch (err) {
        console.error(chalk.red(`Error: ${err.message}`));
        process.exit(1);
      }
    });

  threadCmd
    .command('release <path>')
    .description('Release a claimed thread back to open')
    .option('-v, --vault <path>', 'Vault path')
    .option('-a, --actor <name>', 'Agent name', agentName)
    .option('--reason <reason>', 'Why you are releasing')
    .action(async (threadPath, opts) => {
      try {
        const vaultPath = resolveVaultPath(opts.vault);
        const { thread } = await import('../dist/workgraph/index.js');
        thread.release(vaultPath, threadPath, opts.actor, opts.reason);
        console.log(chalk.green(`✓ Released: ${threadPath}`));
      } catch (err) {
        console.error(chalk.red(`Error: ${err.message}`));
        process.exit(1);
      }
    });

  threadCmd
    .command('done <path>')
    .description('Mark thread complete')
    .option('-v, --vault <path>', 'Vault path')
    .option('-a, --actor <name>', 'Agent name', agentName)
    .option('-o, --output <text>', 'Output/result summary')
    .action(async (threadPath, opts) => {
      try {
        const vaultPath = resolveVaultPath(opts.vault);
        const { thread } = await import('../dist/workgraph/index.js');
        thread.done(vaultPath, threadPath, opts.actor, opts.output);
        console.log(chalk.green(`✓ Done: ${threadPath}`));
      } catch (err) {
        console.error(chalk.red(`Error: ${err.message}`));
        process.exit(1);
      }
    });

  threadCmd
    .command('block <path>')
    .description('Mark thread blocked on a dependency')
    .option('-v, --vault <path>', 'Vault path')
    .option('-a, --actor <name>', 'Agent name', agentName)
    .requiredOption('-b, --blocked-by <dep>', 'What is blocking this thread')
    .option('--reason <reason>', 'Why it is blocked')
    .action(async (threadPath, opts) => {
      try {
        const vaultPath = resolveVaultPath(opts.vault);
        const { thread } = await import('../dist/workgraph/index.js');
        thread.block(vaultPath, threadPath, opts.actor, opts.blockedBy, opts.reason);
        console.log(chalk.red(`⊘ Blocked: ${threadPath}`));
        console.log(`  Blocked by: ${opts.blockedBy}`);
      } catch (err) {
        console.error(chalk.red(`Error: ${err.message}`));
        process.exit(1);
      }
    });

  threadCmd
    .command('unblock <path>')
    .description('Unblock a thread')
    .option('-v, --vault <path>', 'Vault path')
    .option('-a, --actor <name>', 'Agent name', agentName)
    .action(async (threadPath, opts) => {
      try {
        const vaultPath = resolveVaultPath(opts.vault);
        const { thread } = await import('../dist/workgraph/index.js');
        thread.unblock(vaultPath, threadPath, opts.actor);
        console.log(chalk.green(`✓ Unblocked: ${threadPath}`));
      } catch (err) {
        console.error(chalk.red(`Error: ${err.message}`));
        process.exit(1);
      }
    });

  threadCmd
    .command('decompose <path>')
    .description('Break a thread into sub-threads')
    .option('-v, --vault <path>', 'Vault path')
    .option('-a, --actor <name>', 'Agent name', agentName)
    .requiredOption('--sub <specs...>', 'Sub-threads as "title|goal" pairs')
    .action(async (threadPath, opts) => {
      try {
        const vaultPath = resolveVaultPath(opts.vault);
        const { thread } = await import('../dist/workgraph/index.js');
        const subs = opts.sub.map(spec => {
          const [title, ...goalParts] = spec.split('|');
          return { title: title.trim(), goal: goalParts.join('|').trim() || title.trim() };
        });
        const children = thread.decompose(vaultPath, threadPath, subs, opts.actor);
        console.log(chalk.green(`✓ Decomposed ${threadPath} into ${children.length} sub-threads:`));
        for (const c of children) {
          console.log(`  → ${c.path}`);
        }
      } catch (err) {
        console.error(chalk.red(`Error: ${err.message}`));
        process.exit(1);
      }
    });

  // =========================================================================
  // primitive
  // =========================================================================
  const primitiveCmd = program
    .command('primitive')
    .description('Manage workgraph primitive types (define new types, list, create)');

  primitiveCmd
    .command('define <name>')
    .description('Define a new primitive type that agents can instantiate')
    .requiredOption('-d, --description <desc>', 'What this type represents')
    .option('-v, --vault <path>', 'Vault path')
    .option('-a, --actor <name>', 'Agent name', agentName)
    .option('--fields <specs...>', 'Field definitions as "name:type" (types: string, number, boolean, list, date, ref)')
    .option('--dir <directory>', 'Storage directory name')
    .action(async (name, opts) => {
      try {
        const vaultPath = resolveVaultPath(opts.vault);
        const { registry } = await import('../dist/workgraph/index.js');
        const fields = {};
        if (opts.fields) {
          for (const spec of opts.fields) {
            const [fieldName, fieldType = 'string'] = spec.split(':');
            fields[fieldName.trim()] = { type: fieldType.trim() };
          }
        }
        const typeDef = registry.defineType(vaultPath, name, opts.description, fields, opts.actor, opts.dir);
        console.log(chalk.green(`✓ Defined type: ${typeDef.name}`));
        console.log(`  Directory: ${typeDef.directory}/`);
        console.log(`  Fields:    ${Object.keys(typeDef.fields).join(', ')}`);
      } catch (err) {
        console.error(chalk.red(`Error: ${err.message}`));
        process.exit(1);
      }
    });

  primitiveCmd
    .command('list')
    .description('List all registered primitive types')
    .option('-v, --vault <path>', 'Vault path')
    .option('--json', 'Output as JSON')
    .action(async (opts) => {
      try {
        const vaultPath = resolveVaultPath(opts.vault);
        const { registry } = await import('../dist/workgraph/index.js');
        const types = registry.listTypes(vaultPath);

        if (opts.json) {
          console.log(JSON.stringify(types, null, 2));
          return;
        }

        for (const t of types) {
          const badge = t.builtIn ? chalk.dim('[built-in]') : chalk.cyan(`[${t.createdBy}]`);
          console.log(`  ${chalk.bold(t.name)} ${badge}`);
          console.log(chalk.dim(`    ${t.description}`));
          console.log(chalk.dim(`    dir: ${t.directory}/  fields: ${Object.keys(t.fields).join(', ')}`));
        }
        console.log(chalk.dim(`\n${types.length} type(s) — ${types.filter(t => !t.builtIn).length} agent-defined`));
      } catch (err) {
        console.error(chalk.red(`Error: ${err.message}`));
        process.exit(1);
      }
    });

  primitiveCmd
    .command('create <type> <title>')
    .description('Create an instance of any primitive type')
    .option('-v, --vault <path>', 'Vault path')
    .option('-a, --actor <name>', 'Agent name', agentName)
    .option('--set <fields...>', 'Set fields as "key=value" pairs')
    .option('--body <text>', 'Markdown body content', '')
    .action(async (type, title, opts) => {
      try {
        const vaultPath = resolveVaultPath(opts.vault);
        const { store } = await import('../dist/workgraph/index.js');
        const fields = { title };
        if (opts.set) {
          for (const pair of opts.set) {
            const eqIdx = pair.indexOf('=');
            if (eqIdx === -1) continue;
            const key = pair.slice(0, eqIdx).trim();
            let val = pair.slice(eqIdx + 1).trim();
            if (val.includes(',')) val = val.split(',').map(s => s.trim());
            fields[key] = val;
          }
        }
        const inst = store.create(vaultPath, type, fields, opts.body, opts.actor);
        console.log(chalk.green(`✓ Created ${type}: ${inst.path}`));
      } catch (err) {
        console.error(chalk.red(`Error: ${err.message}`));
        process.exit(1);
      }
    });

  // =========================================================================
  // ledger
  // =========================================================================
  const ledgerCmd = program
    .command('ledger')
    .description('View the workgraph audit trail');

  ledgerCmd
    .command('show')
    .description('Show recent ledger entries')
    .option('-v, --vault <path>', 'Vault path')
    .option('-n, --count <n>', 'Number of entries', '20')
    .option('--actor <name>', 'Filter by actor')
    .option('--json', 'Output as JSON')
    .action(async (opts) => {
      try {
        const vaultPath = resolveVaultPath(opts.vault);
        const { ledger } = await import('../dist/workgraph/index.js');
        let entries = ledger.recent(vaultPath, parseInt(opts.count));
        if (opts.actor) entries = entries.filter(e => e.actor === opts.actor);

        if (opts.json) {
          console.log(JSON.stringify(entries, null, 2));
          return;
        }

        if (entries.length === 0) {
          console.log(chalk.dim('No ledger entries.'));
          return;
        }

        const opColor = { create: 'green', claim: 'yellow', release: 'blue', done: 'green', block: 'red', unblock: 'cyan', cancel: 'dim', update: 'white', delete: 'red', define: 'magenta', decompose: 'cyan' };
        for (const e of entries) {
          const time = new Date(e.ts).toLocaleString();
          const colorFn = chalk[opColor[e.op]] || chalk.white;
          const data = e.data ? chalk.dim(` ${JSON.stringify(e.data)}`) : '';
          console.log(`  ${chalk.dim(time)} ${colorFn(e.op.padEnd(10))} ${e.actor.padEnd(15)} ${e.target}${data}`);
        }
      } catch (err) {
        console.error(chalk.red(`Error: ${err.message}`));
        process.exit(1);
      }
    });

  ledgerCmd
    .command('history <path>')
    .description('Show full history of a specific file')
    .option('-v, --vault <path>', 'Vault path')
    .action(async (targetPath, opts) => {
      try {
        const vaultPath = resolveVaultPath(opts.vault);
        const { ledger } = await import('../dist/workgraph/index.js');
        const history = ledger.historyOf(vaultPath, targetPath);

        if (history.length === 0) {
          console.log(chalk.dim(`No history for ${targetPath}`));
          return;
        }

        console.log(chalk.bold(`History: ${targetPath}`));
        console.log(chalk.dim('─'.repeat(50)));
        for (const e of history) {
          const time = new Date(e.ts).toLocaleString();
          const data = e.data ? chalk.dim(` ${JSON.stringify(e.data)}`) : '';
          console.log(`  ${chalk.dim(time)} ${chalk.cyan(e.op)} by ${e.actor}${data}`);
        }
      } catch (err) {
        console.error(chalk.red(`Error: ${err.message}`));
        process.exit(1);
      }
    });

  ledgerCmd
    .command('claims')
    .description('Show all active claims')
    .option('-v, --vault <path>', 'Vault path')
    .action(async (opts) => {
      try {
        const vaultPath = resolveVaultPath(opts.vault);
        const { ledger } = await import('../dist/workgraph/index.js');
        const claims = ledger.allClaims(vaultPath);

        if (claims.size === 0) {
          console.log(chalk.dim('No active claims.'));
          return;
        }

        for (const [target, owner] of claims) {
          console.log(`  ${chalk.yellow(owner.padEnd(20))} → ${target}`);
        }
        console.log(chalk.dim(`\n${claims.size} active claim(s)`));
      } catch (err) {
        console.error(chalk.red(`Error: ${err.message}`));
        process.exit(1);
      }
    });
}

function csv(value) {
  if (!value) return undefined;
  return String(value).split(',').map(s => s.trim()).filter(Boolean);
}

// ─────────────────────────────────────────────────────────────────────────────
// Beautiful, agent-native 'wg' namespace
// ─────────────────────────────────────────────────────────────────────────────

const BOX = {
  topLeft: '╭',
  topRight: '╮',
  bottomLeft: '╰',
  bottomRight: '╯',
  horizontal: '─',
  vertical: '│',
};

const PRIORITY_CONFIG = {
  urgent: { symbol: '🔴', label: 'URGENT' },
  high: { symbol: '🟠', label: 'HIGH' },
  medium: { symbol: '🔵', label: 'MEDIUM' },
  low: { symbol: '⚪', label: 'LOW' },
};

const STATUS_CONFIG = {
  open: { symbol: '○', label: 'Open' },
  active: { symbol: '●', label: 'Active' },
  blocked: { symbol: '⊘', label: 'Blocked' },
  done: { symbol: '✓', label: 'Done' },
  cancelled: { symbol: '✗', label: 'Cancelled' },
};

const OP_COLORS = {
  create: 'green',
  update: 'blue',
  delete: 'red',
  claim: 'yellow',
  release: 'cyan',
  block: 'red',
  unblock: 'green',
  done: 'greenBright',
  cancel: 'gray',
  define: 'magenta',
  decompose: 'cyan',
};

function getAgentName() {
  return process.env.CLAWVAULT_AGENT || os.hostname();
}

function formatRelativeTime(isoTimestamp) {
  const now = Date.now();
  const then = new Date(isoTimestamp).getTime();
  const diffMs = now - then;
  if (diffMs < 0) return 'just now';
  const seconds = Math.floor(diffMs / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);
  const weeks = Math.floor(days / 7);
  if (weeks > 0) return `${weeks}w ago`;
  if (days > 0) return `${days}d ago`;
  if (hours > 0) return `${hours}h ago`;
  if (minutes > 0) return `${minutes}m ago`;
  if (seconds > 0) return `${seconds}s ago`;
  return 'just now';
}

function truncate(str, maxLen) {
  if (str.length <= maxLen) return str;
  return str.slice(0, maxLen - 1) + '…';
}

function stripAnsi(str) {
  return str.replace(/\x1B\[[0-9;]*m/g, '');
}

function drawLine(width, title) {
  if (!title) return BOX.horizontal.repeat(width);
  const titlePadded = ` ${title} `;
  const remaining = width - titlePadded.length - 2;
  const left = Math.floor(remaining / 2);
  const right = remaining - left;
  return BOX.horizontal.repeat(left) + titlePadded + BOX.horizontal.repeat(right);
}

function drawBox(chalk, title, lines, width = 60) {
  const innerWidth = width - 2;
  const output = [];
  output.push(BOX.topLeft + drawLine(innerWidth, title) + BOX.topRight);
  for (const line of lines) {
    const stripped = stripAnsi(line);
    const padding = innerWidth - stripped.length;
    output.push(BOX.vertical + line + ' '.repeat(Math.max(0, padding)) + BOX.vertical);
  }
  output.push(BOX.bottomLeft + BOX.horizontal.repeat(innerWidth) + BOX.bottomRight);
  return output.join('\n');
}

function getGreeting(hour) {
  if (hour < 12) return '☀️  Good morning';
  if (hour < 17) return '🌤️  Good afternoon';
  return '🌙 Good evening';
}

function sortByPriority(threads) {
  const priorityOrder = { urgent: 0, high: 1, medium: 2, low: 3 };
  return [...threads].sort((a, b) => {
    const pa = priorityOrder[String(a.fields.priority || 'medium')] ?? 2;
    const pb = priorityOrder[String(b.fields.priority || 'medium')] ?? 2;
    return pa - pb;
  });
}

function formatThreadLine(chalk, inst, showOwner = true) {
  const status = inst.fields.status;
  const priority = inst.fields.priority || 'medium';
  const title = truncate(String(inst.fields.title || inst.path), 40);
  const owner = inst.fields.owner;
  const statusCfg = STATUS_CONFIG[status] || STATUS_CONFIG.open;
  const priorityCfg = PRIORITY_CONFIG[priority] || PRIORITY_CONFIG.medium;
  const statusColors = { open: 'cyan', active: 'green', blocked: 'red', done: 'gray', cancelled: 'dim' };
  const priorityColors = { urgent: 'red', high: 'yellow', medium: 'blue', low: 'gray' };
  const statusColor = chalk[statusColors[status]] || chalk.white;
  let line = `${statusColor(statusCfg.symbol)} ${priorityCfg.symbol} ${chalk.white(title)}`;
  if (showOwner && owner) {
    line += chalk.dim(` @${owner}`);
  }
  return line;
}

function formatPriority(chalk, priority) {
  const cfg = PRIORITY_CONFIG[priority] || PRIORITY_CONFIG.medium;
  const colors = { urgent: 'red', high: 'yellow', medium: 'blue', low: 'gray' };
  const colorFn = chalk[colors[priority]] || chalk.blue;
  return colorFn(`${cfg.symbol} ${cfg.label}`);
}

function normalizeThreadPath(input) {
  if (input.startsWith('threads/')) return input;
  if (input.endsWith('.md')) return `threads/${input}`;
  return `threads/${input}.md`;
}

function formatError(chalk, what, why, fix) {
  return [
    '',
    chalk.red.bold('✗ Error: ') + chalk.red(what),
    '',
    chalk.dim('Why: ') + why,
    chalk.dim('Fix: ') + chalk.cyan(fix),
    '',
  ].join('\n');
}

function registerWgCommands(program, { chalk, resolveVaultPath }) {
  const wg = program
    .command('wg')
    .description('Workgraph — beautiful, agent-native multi-agent coordination');

  // wg status
  wg.command('status')
    .description('Agent morning briefing with active work, available tasks, and team status')
    .option('-v, --vault <path>', 'Vault path')
    .action(async (opts) => {
      try {
        const vaultPath = resolveVaultPath(opts.vault);
        const { store, ledger } = await import('../dist/workgraph/index.js');
        const agentName = getAgentName();
        const now = new Date();
        const greeting = getGreeting(now.getHours());

        console.log('');
        console.log(chalk.bold.cyan(`${greeting}, ${agentName}!`));
        console.log(chalk.dim(`${now.toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}`));
        console.log('');

        const allThreads = store.list(vaultPath, 'thread');
        const activeThreads = allThreads.filter(t => t.fields.status === 'active');
        const openThreads = allThreads.filter(t => t.fields.status === 'open');
        const blockedThreads = allThreads.filter(t => t.fields.status === 'blocked');
        const myActiveThreads = activeThreads.filter(t => t.fields.owner === agentName);

        if (myActiveThreads.length > 0) {
          const activeLines = myActiveThreads.map(t => formatThreadLine(chalk, t, false));
          console.log(drawBox(chalk, '🔥 Your Active Work', activeLines, 65));
          console.log('');
        }

        if (openThreads.length > 0) {
          const sorted = sortByPriority(openThreads);
          const availableLines = sorted.slice(0, 5).map(t => formatThreadLine(chalk, t, false));
          if (sorted.length > 5) {
            availableLines.push(chalk.dim(`  ... and ${sorted.length - 5} more`));
          }
          console.log(drawBox(chalk, '📋 Available Work', availableLines, 65));
          console.log('');
        }

        if (blockedThreads.length > 0) {
          const blockedLines = blockedThreads.slice(0, 3).map(t => {
            const title = truncate(String(t.fields.title || t.path), 35);
            const deps = t.fields.deps || [];
            const depStr = deps.length > 0 ? chalk.dim(` → ${deps[0]}`) : '';
            const statusColor = chalk.red;
            return `${statusColor(STATUS_CONFIG.blocked.symbol)} ${title}${depStr}`;
          });
          console.log(drawBox(chalk, '⛔ Blocked', blockedLines, 65));
          console.log('');
        }

        const recentEntries = ledger.recent(vaultPath, 8);
        if (recentEntries.length > 0) {
          const activityLines = recentEntries.reverse().map(e => {
            const opColorName = OP_COLORS[e.op] || 'white';
            const opColor = chalk[opColorName] || chalk.white;
            const target = truncate(path.basename(e.target, '.md'), 25);
            const time = formatRelativeTime(e.ts);
            return `${opColor(e.op.padEnd(8))} ${chalk.white(target)} ${chalk.dim(time)}`;
          });
          console.log(drawBox(chalk, '📜 Recent Activity', activityLines, 65));
          console.log('');
        }

        const claims = ledger.allClaims(vaultPath);
        const teamMembers = new Map();
        for (const [target, owner] of claims) {
          const current = teamMembers.get(owner) || [];
          current.push(target);
          teamMembers.set(owner, current);
        }

        if (teamMembers.size > 0) {
          const teamLines = [];
          for (const [member, threads] of teamMembers) {
            const isYou = member === agentName;
            const name = isYou ? chalk.green(`${member} (you)`) : chalk.white(member);
            teamLines.push(`${chalk.cyan('●')} ${name}: ${chalk.dim(`${threads.length} active`)}`);
          }
          console.log(drawBox(chalk, '👥 Team Status', teamLines, 65));
          console.log('');
        }

        const summaryParts = [
          chalk.green(`${activeThreads.length} active`),
          chalk.cyan(`${openThreads.length} open`),
          chalk.red(`${blockedThreads.length} blocked`),
        ];
        console.log(chalk.dim('Summary: ') + summaryParts.join(chalk.dim(' · ')));
        console.log('');
      } catch (err) {
        console.error(chalk.red(`Error: ${err.message}`));
        process.exit(1);
      }
    });

  // wg thread subcommands
  const threadCmd = wg
    .command('thread')
    .description('Thread lifecycle operations');

  threadCmd
    .command('create <title>')
    .description('Create a new thread')
    .option('--goal <goal>', 'What success looks like')
    .option('--priority <priority>', 'urgent | high | medium | low', 'medium')
    .option('--deps <deps>', 'Comma-separated dependency paths')
    .option('--tags <tags>', 'Comma-separated tags')
    .option('-v, --vault <path>', 'Vault path')
    .action(async (title, opts) => {
      try {
        const vaultPath = resolveVaultPath(opts.vault);
        const { thread } = await import('../dist/workgraph/index.js');
        const agentName = getAgentName();
        const goal = opts.goal || `Complete: ${title}`;
        const priority = opts.priority || 'medium';
        const deps = csv(opts.deps) || [];
        const tags = csv(opts.tags) || [];

        const inst = thread.createThread(vaultPath, title, goal, agentName, {
          priority,
          deps,
          tags,
        });

        console.log('');
        console.log(chalk.green.bold('✓ Thread created'));
        console.log('');
        console.log(chalk.dim('  Path:     ') + chalk.white(inst.path));
        console.log(chalk.dim('  Title:    ') + chalk.white(inst.fields.title));
        console.log(chalk.dim('  Goal:     ') + chalk.white(inst.fields.goal));
        console.log(chalk.dim('  Priority: ') + formatPriority(chalk, priority));
        if (deps.length > 0) {
          console.log(chalk.dim('  Deps:     ') + chalk.cyan(deps.join(', ')));
        }
        if (tags.length > 0) {
          console.log(chalk.dim('  Tags:     ') + chalk.magenta(tags.join(', ')));
        }
        console.log('');
        console.log(chalk.dim(`Claim it: ${chalk.cyan(`clawvault wg thread claim ${inst.path}`)}`));
        console.log('');
      } catch (err) {
        console.error(formatError(chalk,
          'Failed to create thread',
          err.message,
          'Check the title is unique and vault path is correct'
        ));
        process.exit(1);
      }
    });

  threadCmd
    .command('list')
    .description('List threads with optional filters')
    .option('--status <status>', 'Filter by status: open | active | blocked | done | cancelled')
    .option('--owner <owner>', 'Filter by owner (use "me" for current agent)')
    .option('--json', 'Output as JSON')
    .option('-v, --vault <path>', 'Vault path')
    .action(async (opts) => {
      try {
        const vaultPath = resolveVaultPath(opts.vault);
        const { store } = await import('../dist/workgraph/index.js');
        let threads = store.list(vaultPath, 'thread');

        if (opts.status) {
          threads = threads.filter(t => t.fields.status === opts.status);
        }

        if (opts.owner) {
          const ownerFilter = opts.owner === 'me' ? getAgentName() : opts.owner;
          threads = threads.filter(t => t.fields.owner === ownerFilter);
        }

        if (opts.json) {
          console.log(JSON.stringify(threads, null, 2));
          return;
        }

        if (threads.length === 0) {
          console.log('');
          console.log(chalk.dim('No threads found matching filters.'));
          console.log('');
          return;
        }

        console.log('');
        console.log(chalk.bold(`Threads (${threads.length})`));
        console.log(chalk.dim('─'.repeat(70)));

        const sorted = sortByPriority(threads);
        for (const t of sorted) {
          const status = t.fields.status;
          const priority = t.fields.priority || 'medium';
          const title = truncate(String(t.fields.title || t.path), 35);
          const owner = t.fields.owner;
          const updated = formatRelativeTime(String(t.fields.updated));

          const statusCfg = STATUS_CONFIG[status] || STATUS_CONFIG.open;
          const priorityCfg = PRIORITY_CONFIG[priority] || PRIORITY_CONFIG.medium;
          const statusColors = { open: 'cyan', active: 'green', blocked: 'red', done: 'gray', cancelled: 'dim' };
          const statusColor = chalk[statusColors[status]] || chalk.white;

          let line = `${statusColor(statusCfg.symbol.padEnd(2))}`;
          line += `${priorityCfg.symbol} `;
          line += chalk.white(title.padEnd(37));
          line += owner ? chalk.cyan(`@${owner}`.padEnd(15)) : ' '.repeat(15);
          line += chalk.dim(updated);

          console.log(line);
        }

        console.log(chalk.dim('─'.repeat(70)));
        console.log('');
      } catch (err) {
        console.error(chalk.red(`Error: ${err.message}`));
        process.exit(1);
      }
    });

  threadCmd
    .command('claim <path>')
    .description('Claim a thread and show work brief')
    .option('-v, --vault <path>', 'Vault path')
    .action(async (threadPath, opts) => {
      try {
        const vaultPath = resolveVaultPath(opts.vault);
        const { thread } = await import('../dist/workgraph/index.js');
        const agentName = getAgentName();
        const normalizedPath = normalizeThreadPath(threadPath);

        const inst = thread.claim(vaultPath, normalizedPath, agentName);

        console.log('');
        console.log(chalk.green.bold('✓ Thread claimed'));
        console.log('');

        const briefLines = [
          chalk.dim('Title:    ') + chalk.white.bold(inst.fields.title),
          chalk.dim('Goal:     ') + chalk.white(inst.fields.goal),
          chalk.dim('Priority: ') + formatPriority(chalk, String(inst.fields.priority || 'medium')),
        ];

        const deps = inst.fields.deps || [];
        if (deps.length > 0) {
          briefLines.push(chalk.dim('Deps:     ') + chalk.cyan(deps.join(', ')));
        }

        const contextRefs = inst.fields.context_refs || [];
        if (contextRefs.length > 0) {
          briefLines.push(chalk.dim('Context:  ') + chalk.magenta(contextRefs.join(', ')));
        }

        console.log(drawBox(chalk, '📋 Work Brief', briefLines, 65));
        console.log('');

        if (inst.body && inst.body.trim()) {
          console.log(chalk.dim('─'.repeat(65)));
          console.log(chalk.dim('Notes:'));
          console.log(inst.body.trim().split('\n').slice(0, 10).join('\n'));
          console.log(chalk.dim('─'.repeat(65)));
          console.log('');
        }

        console.log(chalk.dim(`When done: ${chalk.cyan(`clawvault wg thread done ${normalizedPath}`)}`));
        console.log('');
      } catch (err) {
        console.error(formatError(chalk,
          'Failed to claim thread',
          err.message,
          'Ensure the thread exists and is in "open" status'
        ));
        process.exit(1);
      }
    });

  threadCmd
    .command('done <path>')
    .description('Mark thread as complete')
    .option('--output <output>', 'Completion summary or output')
    .option('-v, --vault <path>', 'Vault path')
    .action(async (threadPath, opts) => {
      try {
        const vaultPath = resolveVaultPath(opts.vault);
        const { thread } = await import('../dist/workgraph/index.js');
        const agentName = getAgentName();
        const normalizedPath = normalizeThreadPath(threadPath);

        const inst = thread.done(vaultPath, normalizedPath, agentName, opts.output);

        console.log('');
        console.log(chalk.green.bold('✓ Thread completed!'));
        console.log('');
        console.log(chalk.dim('  Title:  ') + chalk.white(inst.fields.title));
        console.log(chalk.dim('  Status: ') + chalk.green('done'));
        if (opts.output) {
          console.log(chalk.dim('  Output: ') + chalk.white(truncate(opts.output, 50)));
        }
        console.log('');
        console.log(chalk.dim('Great work! 🎉'));
        console.log('');
      } catch (err) {
        console.error(formatError(chalk,
          'Failed to complete thread',
          err.message,
          'Ensure you own the thread and it is in "active" status'
        ));
        process.exit(1);
      }
    });

  threadCmd
    .command('block <path>')
    .description('Block thread on a dependency')
    .requiredOption('--by <blocker>', 'What is blocking this thread')
    .option('--reason <reason>', 'Additional context')
    .option('-v, --vault <path>', 'Vault path')
    .action(async (threadPath, opts) => {
      try {
        const vaultPath = resolveVaultPath(opts.vault);
        const { thread } = await import('../dist/workgraph/index.js');
        const agentName = getAgentName();
        const normalizedPath = normalizeThreadPath(threadPath);

        const inst = thread.block(vaultPath, normalizedPath, agentName, opts.by, opts.reason);

        console.log('');
        console.log(chalk.yellow.bold('⊘ Thread blocked'));
        console.log('');
        console.log(chalk.dim('  Title:      ') + chalk.white(inst.fields.title));
        console.log(chalk.dim('  Blocked by: ') + chalk.red(opts.by));
        if (opts.reason) {
          console.log(chalk.dim('  Reason:     ') + chalk.white(opts.reason));
        }
        console.log('');
      } catch (err) {
        console.error(formatError(chalk,
          'Failed to block thread',
          err.message,
          'Ensure the thread exists and is in "active" status'
        ));
        process.exit(1);
      }
    });

  threadCmd
    .command('release <path>')
    .description('Release thread back to the pool')
    .option('--reason <reason>', 'Why releasing')
    .option('-v, --vault <path>', 'Vault path')
    .action(async (threadPath, opts) => {
      try {
        const vaultPath = resolveVaultPath(opts.vault);
        const { thread } = await import('../dist/workgraph/index.js');
        const agentName = getAgentName();
        const normalizedPath = normalizeThreadPath(threadPath);

        const inst = thread.release(vaultPath, normalizedPath, agentName, opts.reason);

        console.log('');
        console.log(chalk.cyan.bold('↩ Thread released'));
        console.log('');
        console.log(chalk.dim('  Title:  ') + chalk.white(inst.fields.title));
        console.log(chalk.dim('  Status: ') + chalk.cyan('open'));
        if (opts.reason) {
          console.log(chalk.dim('  Reason: ') + chalk.white(opts.reason));
        }
        console.log('');
        console.log(chalk.dim('Thread is now available for others to claim.'));
        console.log('');
      } catch (err) {
        console.error(formatError(chalk,
          'Failed to release thread',
          err.message,
          'Ensure you own the thread'
        ));
        process.exit(1);
      }
    });

  threadCmd
    .command('decompose <path>')
    .description('Break thread into sub-threads')
    .option('--into <titles...>', 'Sub-thread titles')
    .option('-v, --vault <path>', 'Vault path')
    .action(async (threadPath, opts) => {
      try {
        const vaultPath = resolveVaultPath(opts.vault);
        const { thread, store } = await import('../dist/workgraph/index.js');
        const agentName = getAgentName();
        const normalizedPath = normalizeThreadPath(threadPath);

        if (!opts.into || opts.into.length === 0) {
          console.error(formatError(chalk,
            'Missing --into option',
            'You must specify sub-thread titles',
            'clawvault wg thread decompose <path> --into "sub1" --into "sub2"'
          ));
          process.exit(1);
        }

        const parent = store.read(vaultPath, normalizedPath);
        if (!parent) {
          throw new Error(`Thread not found: ${normalizedPath}`);
        }

        const subthreads = opts.into.map(title => ({
          title,
          goal: `Sub-task of: ${parent.fields.title}`,
        }));

        const created = thread.decompose(vaultPath, normalizedPath, subthreads, agentName);

        console.log('');
        console.log(chalk.green.bold('✓ Thread decomposed'));
        console.log('');
        console.log(chalk.dim('  Parent: ') + chalk.white(parent.fields.title));
        console.log(chalk.dim('  Created sub-threads:'));
        for (const sub of created) {
          console.log(chalk.cyan(`    → ${sub.fields.title}`));
        }
        console.log('');
      } catch (err) {
        console.error(formatError(chalk,
          'Failed to decompose thread',
          err.message,
          'Ensure the thread exists'
        ));
        process.exit(1);
      }
    });

  // wg ledger
  wg.command('ledger')
    .description('View coordination history')
    .option('--last <n>', 'Number of entries to show', '20')
    .option('--actor <actor>', 'Filter by actor (use "me" for current agent)')
    .option('--target <target>', 'Filter by target path substring')
    .option('--json', 'Output as JSON')
    .option('-v, --vault <path>', 'Vault path')
    .action(async (opts) => {
      try {
        const vaultPath = resolveVaultPath(opts.vault);
        const { ledger } = await import('../dist/workgraph/index.js');
        let entries = ledger.readAll(vaultPath);

        if (opts.actor) {
          const actorFilter = opts.actor === 'me' ? getAgentName() : opts.actor;
          entries = entries.filter(e => e.actor === actorFilter);
        }

        if (opts.target) {
          entries = entries.filter(e => e.target.includes(opts.target));
        }

        const limit = parseInt(opts.last) || 20;
        entries = entries.slice(-limit);

        if (opts.json) {
          console.log(JSON.stringify(entries, null, 2));
          return;
        }

        if (entries.length === 0) {
          console.log('');
          console.log(chalk.dim('No ledger entries found.'));
          console.log('');
          return;
        }

        console.log('');
        console.log(chalk.bold(`Ledger (last ${entries.length} entries)`));
        console.log(chalk.dim('─'.repeat(80)));

        for (const entry of entries.reverse()) {
          const opColorName = OP_COLORS[entry.op] || 'white';
          const opColor = chalk[opColorName] || chalk.white;
          const time = formatRelativeTime(entry.ts);
          const target = truncate(entry.target, 30);
          const actor = entry.actor;

          let line = chalk.dim(time.padEnd(10));
          line += opColor(entry.op.toUpperCase().padEnd(10));
          line += chalk.white(target.padEnd(32));
          line += chalk.cyan(`@${actor}`);

          console.log(line);

          if (entry.data && Object.keys(entry.data).length > 0) {
            const dataStr = JSON.stringify(entry.data);
            console.log(chalk.dim(`           ${truncate(dataStr, 68)}`));
          }
        }

        console.log(chalk.dim('─'.repeat(80)));
        console.log('');
      } catch (err) {
        console.error(chalk.red(`Error: ${err.message}`));
        process.exit(1);
      }
    });

  // wg define
  wg.command('define <type>')
    .description('Define a new primitive type')
    .option('--fields <fields>', 'Comma-separated field definitions (name:type)')
    .option('--dir <directory>', 'Custom directory for instances')
    .option('--description <desc>', 'Type description')
    .option('-v, --vault <path>', 'Vault path')
    .action(async (typeName, opts) => {
      try {
        const vaultPath = resolveVaultPath(opts.vault);
        const { registry } = await import('../dist/workgraph/index.js');
        const agentName = getAgentName();
        const description = opts.description || `Custom type: ${typeName}`;

        const fields = {};
        if (opts.fields) {
          const fieldPairs = opts.fields.split(',');
          for (const pair of fieldPairs) {
            const [name, type] = pair.split(':').map(s => s.trim());
            if (name && type) {
              fields[name] = { type };
            }
          }
        }

        const typeDef = registry.defineType(
          vaultPath,
          typeName,
          description,
          fields,
          agentName,
          opts.dir
        );

        console.log('');
        console.log(chalk.green.bold('✓ Type defined'));
        console.log('');
        console.log(chalk.dim('  Name:      ') + chalk.magenta(typeDef.name));
        console.log(chalk.dim('  Directory: ') + chalk.white(typeDef.directory));
        console.log(chalk.dim('  Fields:'));
        for (const [fieldName, fieldDef] of Object.entries(typeDef.fields)) {
          const required = fieldDef.required ? chalk.red('*') : ' ';
          console.log(chalk.dim(`    ${required} ${fieldName}: ${fieldDef.type}`));
        }
        console.log('');
        console.log(chalk.dim(`Create instances: ${chalk.cyan(`clawvault wg create ${typeDef.name} "title"`)}`));
        console.log('');
      } catch (err) {
        console.error(formatError(chalk,
          'Failed to define type',
          err.message,
          'Ensure the type name is unique and not a built-in type'
        ));
        process.exit(1);
      }
    });

  // wg types
  wg.command('types')
    .description('List all primitive types with their fields')
    .option('--json', 'Output as JSON')
    .option('-v, --vault <path>', 'Vault path')
    .action(async (opts) => {
      try {
        const vaultPath = resolveVaultPath(opts.vault);
        const { registry } = await import('../dist/workgraph/index.js');
        const types = registry.listTypes(vaultPath);

        if (opts.json) {
          console.log(JSON.stringify(types, null, 2));
          return;
        }

        console.log('');
        console.log(chalk.bold(`Primitive Types (${types.length})`));
        console.log(chalk.dim('─'.repeat(70)));

        for (const typeDef of types) {
          const builtInBadge = typeDef.builtIn ? chalk.cyan(' [built-in]') : chalk.magenta(' [custom]');
          console.log('');
          console.log(chalk.white.bold(typeDef.name) + builtInBadge);
          console.log(chalk.dim(`  ${typeDef.description}`));
          console.log(chalk.dim(`  Directory: ${typeDef.directory}/`));
          console.log(chalk.dim('  Fields:'));

          const fieldEntries = Object.entries(typeDef.fields);
          for (const [fieldName, fieldDef] of fieldEntries) {
            const required = fieldDef.required ? chalk.red('*') : ' ';
            const defaultVal = fieldDef.default !== undefined ? chalk.dim(` = ${JSON.stringify(fieldDef.default)}`) : '';
            const desc = fieldDef.description ? chalk.dim(` — ${fieldDef.description}`) : '';
            console.log(`    ${required} ${chalk.cyan(fieldName)}: ${fieldDef.type}${defaultVal}${desc}`);
          }
        }

        console.log('');
        console.log(chalk.dim('─'.repeat(70)));
        console.log('');
      } catch (err) {
        console.error(chalk.red(`Error: ${err.message}`));
        process.exit(1);
      }
    });

  // wg create
  wg.command('create <type> <title>')
    .description('Create any primitive instance')
    .option('--body <body>', 'Markdown body content')
    .option('-v, --vault <path>', 'Vault path')
    .allowUnknownOption(true)
    .action(async (typeName, title, opts) => {
      try {
        const vaultPath = resolveVaultPath(opts.vault);
        const { store } = await import('../dist/workgraph/index.js');
        const agentName = getAgentName();
        const body = opts.body || '';

        const fields = { title };
        const knownOptions = new Set(['body', 'vault']);
        for (const [key, value] of Object.entries(opts)) {
          if (!knownOptions.has(key) && value !== undefined) {
            fields[key] = value;
          }
        }

        const inst = store.create(vaultPath, typeName, fields, body, agentName);

        console.log('');
        console.log(chalk.green.bold(`✓ ${typeName} created`));
        console.log('');
        console.log(chalk.dim('  Path:  ') + chalk.white(inst.path));
        console.log(chalk.dim('  Title: ') + chalk.white(inst.fields.title));
        console.log('');
      } catch (err) {
        console.error(formatError(chalk,
          `Failed to create ${typeName}`,
          err.message,
          `Ensure the type "${typeName}" exists. Run: clawvault wg types`
        ));
        process.exit(1);
      }
    });

  // wg board
  wg.command('board')
    .description('Terminal kanban board view')
    .option('--width <width>', 'Terminal width override')
    .option('-v, --vault <path>', 'Vault path')
    .action(async (opts) => {
      try {
        const vaultPath = resolveVaultPath(opts.vault);
        const { store } = await import('../dist/workgraph/index.js');
        const threads = store.list(vaultPath, 'thread');
        const termWidth = opts.width ? parseInt(opts.width) : (process.stdout.columns || 120);

        const columns = {
          open: [],
          active: [],
          blocked: [],
          done: [],
          cancelled: [],
        };

        for (const t of threads) {
          const status = t.fields.status;
          if (columns[status]) {
            columns[status].push(t);
          }
        }

        for (const status of Object.keys(columns)) {
          columns[status] = sortByPriority(columns[status]);
        }

        const visibleStatuses = ['open', 'active', 'blocked', 'done'];
        const colWidth = Math.floor((termWidth - visibleStatuses.length - 1) / visibleStatuses.length);
        const cardWidth = colWidth - 4;

        console.log('');
        console.log(chalk.bold.cyan('╔' + '═'.repeat(termWidth - 2) + '╗'));
        console.log(chalk.bold.cyan('║') + chalk.bold(' WORKGRAPH BOARD').padEnd(termWidth - 2) + chalk.bold.cyan('║'));
        console.log(chalk.bold.cyan('╚' + '═'.repeat(termWidth - 2) + '╝'));
        console.log('');

        let headerLine = '';
        const statusColors = { open: 'cyan', active: 'green', blocked: 'red', done: 'gray', cancelled: 'dim' };
        for (const status of visibleStatuses) {
          const cfg = STATUS_CONFIG[status];
          const header = `${cfg.symbol} ${cfg.label} (${columns[status].length})`;
          const padded = header.padEnd(colWidth);
          const colorFn = chalk[statusColors[status]] || chalk.white;
          headerLine += colorFn(padded);
        }
        console.log(headerLine);
        console.log(chalk.dim('─'.repeat(termWidth)));

        const maxRows = Math.max(...visibleStatuses.map(s => columns[s].length), 1);

        for (let row = 0; row < Math.min(maxRows, 15); row++) {
          let line = '';
          for (const status of visibleStatuses) {
            const t = columns[status][row];
            if (t) {
              const priority = t.fields.priority || 'medium';
              const title = truncate(String(t.fields.title || t.path), cardWidth - 4);
              const owner = t.fields.owner;
              const priorityCfg = PRIORITY_CONFIG[priority] || PRIORITY_CONFIG.medium;

              let card = `${priorityCfg.symbol} ${title}`;
              if (owner) {
                card += chalk.dim(` @${truncate(owner, 8)}`);
              }
              line += card.padEnd(colWidth);
            } else {
              line += ' '.repeat(colWidth);
            }
          }
          console.log(line);
        }

        if (maxRows > 15) {
          console.log(chalk.dim(`... and ${maxRows - 15} more rows`));
        }

        console.log('');
        console.log(chalk.dim('─'.repeat(termWidth)));

        const legendParts = Object.entries(PRIORITY_CONFIG).map(([key, cfg]) =>
          `${cfg.symbol} ${cfg.label}`
        );
        console.log(chalk.dim('Priority: ') + legendParts.join(chalk.dim(' · ')));
        console.log('');
      } catch (err) {
        console.error(chalk.red(`Error: ${err.message}`));
        process.exit(1);
      }
    });
}
