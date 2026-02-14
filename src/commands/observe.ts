import * as fs from 'fs';
import * as path from 'path';
import { spawn } from 'child_process';
import type { Command } from 'commander';
import { Observer } from '../observer/observer.js';
import { parseSessionFile } from '../observer/session-parser.js';
import { SessionWatcher } from '../observer/watcher.js';
import { observeActiveSessions } from '../observer/active-session-observer.js';
import { resolveVaultPath } from '../lib/config.js';

export interface ObserveCommandOptions {
  watch?: string;
  threshold?: number;
  reflectThreshold?: number;
  model?: string;
  compress?: string;
  daemon?: boolean;
  vaultPath?: string;
  active?: boolean;
  agent?: string;
  minNew?: number;
  sessionsDir?: string;
  dryRun?: boolean;
}

function parsePositiveInteger(raw: string, optionName: string): number {
  const parsed = Number.parseInt(raw, 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`Invalid ${optionName}: ${raw}`);
  }
  return parsed;
}

function buildDaemonArgs(options: ObserveCommandOptions): string[] {
  const cliPath = process.argv[1];
  if (!cliPath) {
    throw new Error('Unable to resolve CLI script path for daemon mode.');
  }

  const args = [cliPath, 'observe'];
  if (options.watch) {
    args.push('--watch', options.watch);
  }
  if (options.threshold) {
    args.push('--threshold', String(options.threshold));
  }
  if (options.reflectThreshold) {
    args.push('--reflect-threshold', String(options.reflectThreshold));
  }
  if (options.model) {
    args.push('--model', options.model);
  }
  if (options.vaultPath) {
    args.push('--vault', options.vaultPath);
  }

  return args;
}

async function runOneShotCompression(
  observer: Observer,
  sourceFile: string,
  vaultPath: string
): Promise<void> {
  const resolved = path.resolve(sourceFile);
  if (!fs.existsSync(resolved) || !fs.statSync(resolved).isFile()) {
    throw new Error(`Conversation file not found: ${resolved}`);
  }

  const messages = parseSessionFile(resolved);
  await observer.processMessages(messages);

  // Force flush to capture everything
  const { observations, routingSummary } = await observer.flush();

  const datePart = new Date().toISOString().split('T')[0];
  const outputPath = path.join(vaultPath, 'observations', `${datePart}.md`);
  console.log(`Observations updated: ${outputPath}`);
  if (routingSummary) {
    console.log(routingSummary);
  }
}

async function watchSessions(observer: Observer, watchPath: string): Promise<void> {
  const watcher = new SessionWatcher(watchPath, observer);
  await watcher.start();
  console.log(`Watching session updates: ${watchPath}`);

  await new Promise<void>((resolve) => {
    const shutdown = async (): Promise<void> => {
      process.off('SIGINT', onSigInt);
      process.off('SIGTERM', onSigTerm);
      await watcher.stop();
      resolve();
    };

    const onSigInt = (): void => {
      void shutdown();
    };
    const onSigTerm = (): void => {
      void shutdown();
    };

    process.once('SIGINT', onSigInt);
    process.once('SIGTERM', onSigTerm);
  });
}

export async function observeCommand(options: ObserveCommandOptions): Promise<void> {
  if (options.active && (options.watch || options.compress || options.daemon)) {
    throw new Error('--active cannot be combined with --watch, --compress, or --daemon.');
  }

  if (options.compress && options.daemon) {
    throw new Error('--compress cannot be combined with --daemon.');
  }

  const vaultPath = resolveVaultPath({ explicitPath: options.vaultPath });

  if (options.active) {
    const result = await observeActiveSessions({
      vaultPath,
      agentId: options.agent,
      minNewBytes: options.minNew,
      sessionsDir: options.sessionsDir,
      dryRun: options.dryRun,
      threshold: options.threshold,
      reflectThreshold: options.reflectThreshold,
      model: options.model
    });

    if (result.candidateSessions === 0) {
      console.log(`No active sessions crossed threshold (${result.checkedSessions} checked).`);
      return;
    }

    if (result.dryRun) {
      console.log(
        `Dry run: ${result.candidateSessions} session(s) would be observed (${result.totalNewBytes} new bytes).`
      );
      for (const candidate of result.candidates) {
        console.log(
          `- ${candidate.sessionKey} [${candidate.sourceLabel}] Δ${candidate.newBytes}B (threshold ${candidate.thresholdBytes}B)`
        );
      }
      return;
    }

    console.log(
      `Active observation complete: ${result.observedSessions}/${result.candidateSessions} session(s) observed.`
    );
    return;
  }

  const observer = new Observer(vaultPath, {
    tokenThreshold: options.threshold,
    reflectThreshold: options.reflectThreshold,
    model: options.model
  });

  if (options.compress) {
    await runOneShotCompression(observer, options.compress, vaultPath);
    return;
  }

  let watchPath = options.watch ? path.resolve(options.watch) : '';
  if (!watchPath && options.daemon) {
    watchPath = path.join(vaultPath, 'sessions');
  }

  if (!watchPath) {
    throw new Error('Either --watch or --compress must be provided.');
  }

  if (!fs.existsSync(watchPath)) {
    if (options.daemon && !options.watch) {
      fs.mkdirSync(watchPath, { recursive: true });
    } else {
      throw new Error(`Watch path does not exist: ${watchPath}`);
    }
  }

  if (options.daemon) {
    const daemonArgs = buildDaemonArgs({ ...options, watch: watchPath, vaultPath });
    const child = spawn(process.execPath, daemonArgs, {
      detached: true,
      stdio: 'ignore'
    });
    child.unref();
    console.log(`Observer daemon started (pid: ${child.pid})`);
    return;
  }

  await watchSessions(observer, watchPath);
}

export function registerObserveCommand(program: Command): void {
  program
    .command('observe')
    .description('Observe session files and build observational memory')
    .option('--watch <path>', 'Watch session file or directory')
    .option('--active', 'Observe active OpenClaw sessions incrementally')
    .option('--agent <id>', 'OpenClaw agent ID (default: OPENCLAW_AGENT_ID or clawdious)')
    .option('--min-new <bytes>', 'Override minimum new-content threshold in bytes')
    .option('--sessions-dir <path>', 'Override OpenClaw sessions directory')
    .option('--dry-run', 'Show active observation candidates without compressing')
    .option('--threshold <n>', 'Compression token threshold', '30000')
    .option('--reflect-threshold <n>', 'Reflection token threshold', '40000')
    .option('--model <model>', 'LLM model override')
    .option('--compress <file>', 'One-shot compression for a conversation file')
    .option('--daemon', 'Run in detached background mode')
    .option('-v, --vault <path>', 'Vault path')
    .action(async (rawOptions: {
      watch?: string;
      active?: boolean;
      agent?: string;
      minNew?: string;
      sessionsDir?: string;
      dryRun?: boolean;
      threshold: string;
      reflectThreshold: string;
      model?: string;
      compress?: string;
      daemon?: boolean;
      vault?: string;
    }) => {
      await observeCommand({
        watch: rawOptions.watch,
        active: rawOptions.active,
        agent: rawOptions.agent,
        minNew: rawOptions.minNew ? parsePositiveInteger(rawOptions.minNew, 'min-new') : undefined,
        sessionsDir: rawOptions.sessionsDir,
        dryRun: rawOptions.dryRun,
        threshold: parsePositiveInteger(rawOptions.threshold, 'threshold'),
        reflectThreshold: parsePositiveInteger(rawOptions.reflectThreshold, 'reflect-threshold'),
        model: rawOptions.model,
        compress: rawOptions.compress,
        daemon: rawOptions.daemon,
        vaultPath: rawOptions.vault
      });
    });
}
