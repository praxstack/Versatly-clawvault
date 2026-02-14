import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import {
  getTemplate,
  listTemplates,
  registerTemplate,
  type CanvasTemplate
} from './canvas-templates.js';
import type { Canvas } from './canvas-layout.js';
import { createTask, updateTask } from './task-utils.js';
import { buildOrUpdateMemoryGraphIndex } from './memory-graph.js';
import { generateCanvas } from '../commands/canvas.js';

function makeTempDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'clawvault-canvas-templates-'));
}

function canonicalizeCanvas(canvas: Canvas): {
  nodes: Array<Omit<Canvas['nodes'][number], 'id'>>;
  edges: Array<Omit<Canvas['edges'][number], 'id' | 'fromNode' | 'toNode'> & {
    fromNodeIndex: number;
    toNodeIndex: number;
  }>;
} {
  const indexByNodeId = new Map<string, number>();
  canvas.nodes.forEach((node, index) => {
    indexByNodeId.set(node.id, index);
  });

  const nodes = canvas.nodes.map(({ id: _id, ...rest }) => rest);
  const edges = canvas.edges.map(({ id: _id, fromNode, toNode, ...rest }) => ({
    ...rest,
    fromNodeIndex: indexByNodeId.get(fromNode) ?? -1,
    toNodeIndex: indexByNodeId.get(toNode) ?? -1
  }));

  return { nodes, edges };
}

describe('canvas templates', () => {
  let tempDir: string;

  beforeEach(() => {
    tempDir = makeTempDir();
    fs.mkdirSync(path.join(tempDir, '.clawvault'), { recursive: true });
  });

  afterEach(() => {
    fs.rmSync(tempDir, { recursive: true, force: true });
  });

  it('lists built-in templates', () => {
    const templateIds = listTemplates().map((template) => template.id);
    expect(templateIds).toEqual(expect.arrayContaining([
      'default',
      'project-board',
      'brain',
      'sprint'
    ]));
  });

  it('registers and resolves templates from registry', () => {
    const initialCount = listTemplates().length;
    const customTemplateId = `unit-template-${Date.now()}-${Math.floor(Math.random() * 100000)}`;
    const customTemplate: CanvasTemplate = {
      id: customTemplateId,
      name: 'Unit Test Template',
      description: 'Used for template registry tests.',
      generate: () => ({ nodes: [], edges: [] })
    };

    registerTemplate(customTemplate);

    expect(getTemplate(customTemplateId)).toBeDefined();
    expect(listTemplates().length).toBe(initialCount + 1);
  });

  it('generates valid canvas JSON from each built-in template', async () => {
    // Seed graph data to exercise the brain template path.
    fs.mkdirSync(path.join(tempDir, 'projects'), { recursive: true });
    fs.mkdirSync(path.join(tempDir, 'people'), { recursive: true });
    fs.writeFileSync(path.join(tempDir, 'projects', 'alpha.md'), '# Alpha\n[[people/alice]]\n');
    fs.writeFileSync(path.join(tempDir, 'people', 'alice.md'), '# Alice\n[[projects/alpha]]\n');
    await buildOrUpdateMemoryGraphIndex(tempDir, { forceFull: true });

    for (const templateId of ['default', 'project-board', 'brain', 'sprint']) {
      const template = getTemplate(templateId);
      expect(template).toBeDefined();
      const canvas = template!.generate(tempDir, { project: 'alpha' });
      expect(Array.isArray(canvas.nodes)).toBe(true);
      expect(Array.isArray(canvas.edges)).toBe(true);
    }
  });

  it('filters project-board tasks by project', () => {
    const alphaTask = createTask(tempDir, 'Alpha Task', { project: 'alpha', priority: 'high' });
    const betaTask = createTask(tempDir, 'Beta Task', { project: 'beta', priority: 'critical' });
    updateTask(tempDir, betaTask.slug, { status: 'blocked', blocked_by: alphaTask.slug });

    const projectBoard = getTemplate('project-board');
    expect(projectBoard).toBeDefined();
    const canvas = projectBoard!.generate(tempDir, { project: 'alpha' });
    const filePaths = canvas.nodes
      .filter((node) => node.type === 'file' && typeof node.file === 'string')
      .map((node) => node.file as string);

    expect(filePaths).toContain(`tasks/${alphaTask.slug}.md`);
    expect(filePaths).not.toContain(`tasks/${betaTask.slug}.md`);
  });

  it('renders brain template wiki-link edges for linked entities', async () => {
    fs.mkdirSync(path.join(tempDir, 'projects'), { recursive: true });
    fs.mkdirSync(path.join(tempDir, 'decisions'), { recursive: true });
    fs.writeFileSync(path.join(tempDir, 'projects', 'alpha.md'), '# Alpha\n[[decisions/stack-choice]]');
    fs.writeFileSync(path.join(tempDir, 'decisions', 'stack-choice.md'), '# Stack Choice\n[[projects/alpha]]');
    await buildOrUpdateMemoryGraphIndex(tempDir, { forceFull: true });

    const brainTemplate = getTemplate('brain');
    const canvas = brainTemplate!.generate(tempDir, {});
    const wikiEdges = canvas.edges.filter((edge) => edge.label === 'wiki-link');
    expect(wikiEdges.length).toBeGreaterThan(0);
  });

  it('keeps default template output aligned with generateCanvas output', () => {
    createTask(tempDir, 'Default Template Task');

    const defaultTemplate = getTemplate('default');
    expect(defaultTemplate).toBeDefined();

    const fromCommand = generateCanvas(tempDir);
    const fromTemplate = defaultTemplate!.generate(tempDir, {});

    expect(canonicalizeCanvas(fromTemplate)).toEqual(canonicalizeCanvas(fromCommand));
  });
});
