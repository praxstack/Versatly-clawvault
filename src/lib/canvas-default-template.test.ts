import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { generateDefaultCanvas } from './canvas-default-template.js';
import { buildOrUpdateMemoryGraphIndex } from './memory-graph.js';
import { createTask, completeTask, updateTask } from './task-utils.js';
import { ensureLedgerStructure, getObservationPath } from './ledger.js';

function makeTempDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'clawvault-canvas-default-'));
}

function canvasText(canvas: { nodes: Array<{ type: string; text?: string }> }): string {
  return canvas.nodes
    .filter((node) => node.type === 'text')
    .map((node) => node.text ?? '')
    .join('\n');
}

describe('canvas-default-template', () => {
  let tempDir: string;

  beforeEach(() => {
    tempDir = makeTempDir();
    fs.mkdirSync(path.join(tempDir, '.clawvault'), { recursive: true });
  });

  afterEach(() => {
    fs.rmSync(tempDir, { recursive: true, force: true });
  });

  it('generates the default groups and empty edges', () => {
    const canvas = generateDefaultCanvas(tempDir);
    const groupLabels = canvas.nodes
      .filter((node) => node.type === 'group')
      .map((node) => node.label);

    expect(groupLabels).toEqual(expect.arrayContaining(['Vault Status', 'Recent Observations', 'Graph Stats']));
    expect(canvas.edges).toEqual([]);
  });

  it('summarizes task status buckets including truncation for long lists', () => {
    for (let i = 0; i < 7; i += 1) {
      createTask(tempDir, `Open task ${i}`);
    }
    const inProgress = createTask(tempDir, 'In progress task');
    const blocked = createTask(tempDir, 'Blocked task');
    const done = createTask(tempDir, 'Done task');

    updateTask(tempDir, inProgress.slug, { status: 'in-progress' });
    updateTask(tempDir, blocked.slug, { status: 'blocked', blocked_by: 'dependency' });
    completeTask(tempDir, done.slug);

    const text = canvasText(generateDefaultCanvas(tempDir));
    expect(text).toContain('Tasks by Status');
    expect(text).toContain('Total: 10');
    expect(text).toContain('Open (7)');
    expect(text).toContain('In Progress (1)');
    expect(text).toContain('Blocked (1)');
    expect(text).toContain('Done (1)');
    expect(text).toContain('... and 1 more');
  });

  it('shows recent observation titles from heading or first body line', () => {
    ensureLedgerStructure(tempDir);

    const withHeading = getObservationPath(tempDir, '2026-02-14');
    fs.mkdirSync(path.dirname(withHeading), { recursive: true });
    fs.writeFileSync(withHeading, '# Daily Sync\n\nCaptured insights.', 'utf-8');

    const withBodyFallback = getObservationPath(tempDir, '2026-02-15');
    fs.mkdirSync(path.dirname(withBodyFallback), { recursive: true });
    fs.writeFileSync(
      withBodyFallback,
      `---
title: ignored
---

Observation without heading line
`,
      'utf-8'
    );

    const text = canvasText(generateDefaultCanvas(tempDir));
    expect(text).toContain('Total days: 2');
    expect(text).toContain('2026-02-15: Observation without heading line');
    expect(text).toContain('2026-02-14: Daily Sync');
  });

  it('renders graph guidance when index is missing and stats when present', async () => {
    const initial = canvasText(generateDefaultCanvas(tempDir));
    expect(initial).toContain('Graph index not found.');
    expect(initial).toContain('clawvault graph --build');

    fs.mkdirSync(path.join(tempDir, 'projects'), { recursive: true });
    fs.mkdirSync(path.join(tempDir, 'people'), { recursive: true });
    fs.writeFileSync(path.join(tempDir, 'projects', 'alpha.md'), '# Alpha\n[[people/alice]]\n', 'utf-8');
    fs.writeFileSync(path.join(tempDir, 'people', 'alice.md'), '# Alice\n[[projects/alpha]]\n', 'utf-8');
    await buildOrUpdateMemoryGraphIndex(tempDir, { forceFull: true });

    const updated = canvasText(generateDefaultCanvas(tempDir));
    expect(updated).toContain('Graph Stats');
    expect(updated).toContain('Nodes:');
    expect(updated).toContain('Edges:');
    expect(updated).toContain('Node types:');
  });
});
