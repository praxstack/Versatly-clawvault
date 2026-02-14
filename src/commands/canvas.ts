/**
 * Canvas command for ClawVault
 * Generates Obsidian JSON Canvas dashboards using registered templates.
 */

import * as fs from 'fs';
import * as path from 'path';
import type { Canvas } from '../lib/canvas-layout.js';
import {
  getTemplate,
  listTemplates,
  type CanvasTemplate,
  type CanvasTemplateOptions
} from '../lib/canvas-templates.js';

export interface CanvasOptions {
  output?: string;
  template?: string;
  listTemplates?: boolean;
  project?: string;
}

function resolveTemplateOrThrow(templateId: string): CanvasTemplate {
  const template = getTemplate(templateId);
  if (template) {
    return template;
  }

  const available = listTemplates().map((entry) => entry.id).join(', ');
  throw new Error(
    `Unknown canvas template: ${templateId}. Available templates: ${available}`
  );
}

function toTemplateOptions(options: CanvasOptions): CanvasTemplateOptions {
  return {
    project: options.project
  };
}

/**
 * Generate the default dashboard canvas.
 * Kept for compatibility with existing tests and callers.
 */
export function generateCanvas(vaultPath: string): Canvas {
  const defaultTemplate = resolveTemplateOrThrow('default');
  return defaultTemplate.generate(path.resolve(vaultPath), {});
}

function printTemplateList(): void {
  console.log('Available canvas templates:');
  for (const template of listTemplates()) {
    console.log(`- ${template.id}: ${template.name}`);
    console.log(`  ${template.description}`);
  }
}

/**
 * Canvas command handler for CLI.
 */
export async function canvasCommand(
  vaultPath: string,
  options: CanvasOptions = {}
): Promise<void> {
  if (options.listTemplates) {
    printTemplateList();
    return;
  }

  const resolvedPath = path.resolve(vaultPath);
  const outputPath = options.output || path.join(resolvedPath, 'dashboard.canvas');
  const templateId = options.template ?? 'default';
  const template = resolveTemplateOrThrow(templateId);
  const templateOptions = toTemplateOptions(options);
  const canvas = template.generate(resolvedPath, templateOptions);

  fs.writeFileSync(outputPath, JSON.stringify(canvas, null, 2));

  console.log(`✓ Generated canvas dashboard: ${outputPath}`);
  console.log(`  Template: ${template.id}`);
  console.log(`  Nodes: ${canvas.nodes.length}`);
  console.log(`  Edges: ${canvas.edges.length}`);
}
