import * as fs from 'fs';
import * as path from 'path';
import matter from 'gray-matter';
import { spawnSync } from 'child_process';
import { fileURLToPath } from 'url';

export type CompatStatus = 'ok' | 'warn' | 'error';

export interface CompatCheck {
  label: string;
  status: CompatStatus;
  detail?: string;
  hint?: string;
}

export interface CompatReport {
  generatedAt: string;
  checks: CompatCheck[];
  warnings: number;
  errors: number;
}

interface CompatOptions {
  baseDir?: string;
}

const REQUIRED_HOOK_EVENTS = ['gateway:startup', 'command:new', 'session:start'];

function readOptionalFile(filePath: string): string | null {
  try {
    if (!fs.existsSync(filePath)) return null;
    return fs.readFileSync(filePath, 'utf-8');
  } catch {
    return null;
  }
}

function resolveProjectFile(relativePath: string, baseDir?: string): string {
  const fromBaseDir = path.resolve(baseDir ?? process.cwd(), relativePath);
  if (fs.existsSync(fromBaseDir)) {
    return fromBaseDir;
  }
  const fromCwd = path.resolve(process.cwd(), relativePath);
  if (fs.existsSync(fromCwd)) {
    return fromCwd;
  }
  return fileURLToPath(new URL(`../../${relativePath}`, import.meta.url));
}

function checkOpenClawCli(): CompatCheck {
  const result = spawnSync('openclaw', ['--version'], { stdio: 'ignore' });
  if (result.error) {
    return {
      label: 'openclaw CLI available',
      status: 'warn',
      detail: 'openclaw binary not found',
      hint: 'Install OpenClaw CLI to enable hook runtime validation.'
    };
  }
  return { label: 'openclaw CLI available', status: 'ok' };
}

function checkPackageHookRegistration(options: CompatOptions): CompatCheck {
  const packageRaw = readOptionalFile(resolveProjectFile('package.json', options.baseDir));
  if (!packageRaw) {
    return {
      label: 'package hook registration',
      status: 'error',
      detail: 'package.json not found'
    };
  }

  try {
    const parsed = JSON.parse(packageRaw) as { openclaw?: { hooks?: string[] } };
    const registeredHooks = parsed.openclaw?.hooks ?? [];
    if (registeredHooks.includes('./hooks/clawvault')) {
      return {
        label: 'package hook registration',
        status: 'ok',
        detail: './hooks/clawvault'
      };
    }
    return {
      label: 'package hook registration',
      status: 'error',
      detail: 'Missing ./hooks/clawvault in package openclaw.hooks'
    };
  } catch (err: any) {
    return {
      label: 'package hook registration',
      status: 'error',
      detail: err?.message || 'Unable to parse package.json'
    };
  }
}

function checkHookManifest(options: CompatOptions): CompatCheck {
  const hookRaw = readOptionalFile(resolveProjectFile('hooks/clawvault/HOOK.md', options.baseDir));
  if (!hookRaw) {
    return {
      label: 'hook manifest',
      status: 'error',
      detail: 'HOOK.md not found'
    };
  }

  try {
    const parsed = matter(hookRaw);
    const openclaw = (parsed.data?.metadata as { openclaw?: { events?: string[] } } | undefined)?.openclaw;
    const events = Array.isArray(openclaw?.events) ? openclaw?.events ?? [] : [];
    const missingEvents = REQUIRED_HOOK_EVENTS.filter((event) => !events.includes(event));
    if (missingEvents.length === 0) {
      return {
        label: 'hook manifest events',
        status: 'ok',
        detail: events.join(', ')
      };
    }
    return {
      label: 'hook manifest events',
      status: 'error',
      detail: `Missing events: ${missingEvents.join(', ')}`
    };
  } catch (err: any) {
    return {
      label: 'hook manifest events',
      status: 'error',
      detail: err?.message || 'Unable to parse HOOK.md frontmatter'
    };
  }
}

function checkHookHandlerSafety(options: CompatOptions): CompatCheck {
  const handlerRaw = readOptionalFile(resolveProjectFile('hooks/clawvault/handler.js', options.baseDir));
  if (!handlerRaw) {
    return {
      label: 'hook handler script',
      status: 'error',
      detail: 'handler.js not found'
    };
  }

  const usesExecFileSync = handlerRaw.includes('execFileSync');
  const usesExecSync = /\bexecSync\b/.test(handlerRaw);
  if (!usesExecFileSync || usesExecSync) {
    return {
      label: 'hook handler safety',
      status: 'warn',
      detail: 'Expected execFileSync-only execution path',
      hint: 'Prefer execFileSync (no shell) and avoid execSync.'
    };
  }

  return { label: 'hook handler safety', status: 'ok' };
}

function checkSkillMetadata(options: CompatOptions): CompatCheck {
  const skillRaw = readOptionalFile(resolveProjectFile('SKILL.md', options.baseDir));
  if (!skillRaw) {
    return {
      label: 'skill metadata',
      status: 'warn',
      detail: 'SKILL.md not found',
      hint: 'Ensure SKILL.md is present for OpenClaw skill distribution.'
    };
  }

  const hasOpenClawMetadata = /"openclaw"\s*:/.test(skillRaw);
  if (!hasOpenClawMetadata) {
    return {
      label: 'skill metadata',
      status: 'warn',
      detail: 'Missing metadata.openclaw in SKILL.md'
    };
  }

  return { label: 'skill metadata', status: 'ok' };
}

export function checkOpenClawCompatibility(options: CompatOptions = {}): CompatReport {
  const checks = [
    checkOpenClawCli(),
    checkPackageHookRegistration(options),
    checkHookManifest(options),
    checkHookHandlerSafety(options),
    checkSkillMetadata(options)
  ];

  const warnings = checks.filter((check) => check.status === 'warn').length;
  const errors = checks.filter((check) => check.status === 'error').length;

  return {
    generatedAt: new Date().toISOString(),
    checks,
    warnings,
    errors
  };
}

function formatCompatibilityReport(report: CompatReport): string {
  const lines: string[] = [];
  lines.push('OpenClaw Compatibility Report');
  lines.push('-'.repeat(34));
  lines.push(`Generated: ${report.generatedAt}`);
  lines.push('');

  for (const check of report.checks) {
    const prefix = check.status === 'ok'
      ? '✓'
      : check.status === 'warn'
        ? '⚠'
        : '✗';
    lines.push(`${prefix} ${check.label}${check.detail ? ` — ${check.detail}` : ''}`);
    if (check.hint) {
      lines.push(`  ${check.hint}`);
    }
  }

  lines.push('');
  lines.push(`Warnings: ${report.warnings}`);
  lines.push(`Errors: ${report.errors}`);
  return lines.join('\n');
}

export async function compatCommand(options: { json?: boolean } = {}): Promise<void> {
  const report = checkOpenClawCompatibility();
  if (options.json) {
    console.log(JSON.stringify(report, null, 2));
    return;
  }
  console.log(formatCompatibilityReport(report));
}
