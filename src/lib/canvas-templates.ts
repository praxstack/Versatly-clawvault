import type { Canvas } from './canvas-layout.js';
import { defaultCanvasTemplate } from './canvas-default-template.js';
import { projectBoardCanvasTemplate } from './canvas-project-board-template.js';
import { brainCanvasTemplate } from './canvas-brain-template.js';
import { sprintCanvasTemplate } from './canvas-sprint-template.js';

export interface CanvasTemplateOptions {
  project?: string;
  dateRange?: { from: string; to: string };
  width?: number;
  height?: number;
}

export interface CanvasTemplate {
  id: string;
  name: string;
  description: string;
  generate(vaultPath: string, options: CanvasTemplateOptions): Canvas;
}

const templateRegistry = new Map<string, CanvasTemplate>();

function assertTemplateShape(template: CanvasTemplate): void {
  if (!template.id || !template.id.trim()) {
    throw new Error('Canvas template id is required');
  }
  if (!template.name || !template.name.trim()) {
    throw new Error(`Canvas template "${template.id}" must define a name`);
  }
  if (!template.description || !template.description.trim()) {
    throw new Error(`Canvas template "${template.id}" must define a description`);
  }
}

export function registerTemplate(template: CanvasTemplate): void {
  assertTemplateShape(template);
  const normalizedId = template.id.trim();
  if (templateRegistry.has(normalizedId)) {
    throw new Error(`Canvas template already registered: ${normalizedId}`);
  }
  templateRegistry.set(normalizedId, { ...template, id: normalizedId });
}

export function getTemplate(id: string): CanvasTemplate | undefined {
  return templateRegistry.get(id);
}

export function listTemplates(): CanvasTemplate[] {
  return [...templateRegistry.values()]
    .sort((left, right) => left.id.localeCompare(right.id));
}

function ensureBuiltInTemplatesRegistered(): void {
  const builtIns = [
    defaultCanvasTemplate,
    projectBoardCanvasTemplate,
    brainCanvasTemplate,
    sprintCanvasTemplate
  ];

  for (const template of builtIns) {
    if (!templateRegistry.has(template.id)) {
      templateRegistry.set(template.id, template);
    }
  }
}

ensureBuiltInTemplatesRegistered();
