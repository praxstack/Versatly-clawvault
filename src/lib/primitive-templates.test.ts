import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import {
  buildFrontmatterFromTemplate,
  buildTemplateIndex,
  getTemplateFieldNames,
  listTemplateDefinitions,
  loadSchemaTemplateDefinition,
  loadTemplateDefinition,
  normalizeTemplateName,
  parseTemplateDefinition,
  renderDocumentFromTemplate,
  type PrimitiveTemplateDefinition,
  validateFrontmatter,
} from './primitive-templates.js';

function makeTempDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'clawvault-primitive-templates-'));
}

function writeTemplate(dir: string, name: string, content: string): string {
  fs.mkdirSync(dir, { recursive: true });
  const filePath = path.join(dir, `${name}.md`);
  fs.writeFileSync(filePath, content, 'utf-8');
  return filePath;
}

describe('primitive-templates', () => {
  let tempDir: string;

  beforeEach(() => {
    tempDir = makeTempDir();
  });

  afterEach(() => {
    fs.rmSync(tempDir, { recursive: true, force: true });
  });

  it('normalizes names and parses schema templates', () => {
    const parsed = parseTemplateDefinition(
      `---
primitive: task
description: Task schema
fields:
  status:
    type: string
    default: open
  estimate:
    default: 2
---
# {{title}}
`,
      ' task.md '
    );

    expect(normalizeTemplateName(' decision.md ')).toBe('decision');
    expect(parsed.name).toBe('task');
    expect(parsed.format).toBe('schema');
    expect(parsed.primitive).toBe('task');
    expect(parsed.description).toBe('Task schema');
    expect(parsed.fields.status).toMatchObject({ type: 'string', default: 'open' });
    expect(parsed.fields.estimate).toMatchObject({ type: 'number', default: 2 });
  });

  it('falls back to legacy format when schema frontmatter is absent', () => {
    const parsed = parseTemplateDefinition(
      `---
description: Legacy definition
status: open
count: 1
labels:
  - a
metadata:
  owner: alice
---
Body
`,
      'legacy-template.md'
    );

    expect(parsed.format).toBe('legacy');
    expect(parsed.primitive).toBe('legacy-template');
    expect(parsed.description).toBe('Legacy definition');
    expect(parsed.fields.status).toMatchObject({ type: 'string', default: 'open' });
    expect(parsed.fields.count).toMatchObject({ type: 'number', default: 1 });
    expect(parsed.fields.labels).toMatchObject({ type: 'string[]' });
    expect(parsed.fields.metadata).toMatchObject({ type: 'object' });
  });

  it('indexes templates with vault overrides and can load/list definitions', () => {
    const builtinDir = path.join(tempDir, 'builtin');
    const vaultPath = path.join(tempDir, 'vault');
    const vaultTemplatesDir = path.join(vaultPath, 'templates');

    writeTemplate(
      builtinDir,
      'task',
      `---
primitive: task
fields:
  status:
    default: open
---
Builtin task
`
    );
    writeTemplate(
      builtinDir,
      'note',
      `---
description: legacy note
kind: note
---
Builtin note
`
    );
    const vaultTaskPath = writeTemplate(
      vaultTemplatesDir,
      'task',
      `---
primitive: task
fields:
  status:
    default: blocked
  owner:
    required: true
---
Vault task
`
    );

    const index = buildTemplateIndex({
      vaultPath,
      builtinDir,
      ignoreBuiltinNames: new Set(['note']),
    });

    expect(index.get('task')).toBe(vaultTaskPath);
    expect(index.has('note')).toBe(false);

    const task = loadTemplateDefinition('task', { vaultPath, builtinDir });
    expect(task?.format).toBe('schema');
    expect(task?.fields.status.default).toBe('blocked');

    const note = loadTemplateDefinition('note', { vaultPath, builtinDir });
    expect(note?.format).toBe('legacy');
    expect(loadSchemaTemplateDefinition('note', { vaultPath, builtinDir })).toBeNull();

    const listed = listTemplateDefinitions({ vaultPath, builtinDir });
    expect(listed.map((entry) => entry.name)).toEqual(['note', 'task']);
    expect(listed.find((entry) => entry.name === 'task')?.path).toBe(vaultTaskPath);

    expect(getTemplateFieldNames('task', { vaultPath, builtinDir })).toEqual(['status', 'owner']);
    expect(getTemplateFieldNames('missing', { vaultPath, builtinDir })).toEqual([]);
  });

  it('builds and prunes frontmatter with interpolation and overrides', () => {
    const definition: PrimitiveTemplateDefinition = {
      name: 'task',
      primitive: 'task',
      description: 'Task',
      format: 'schema',
      body: 'Body',
      fields: {
        status: { type: 'string', default: 'open' },
        title: { type: 'string', default: '{{title}}' },
        owner: { type: 'string', default: '{{owner}}' },
        labels: { type: 'string[]', default: ['{{type}}', '{{tag}}'] },
        metadata: { type: 'object', default: { date: '{{date}}', by: '{{owner}}' } },
        empty: { type: 'string', default: '' },
      },
    };

    const frontmatter = buildFrontmatterFromTemplate(
      definition,
      {
        title: 'Ship API',
        type: 'task',
        date: '2026-02-14',
        datetime: '2026-02-14T00:00:00.000Z',
        owner: 'alice',
        tag: 'platform',
      },
      {
        status: 'blocked',
        owner: null,
      },
      { pruneEmpty: true }
    );

    expect(frontmatter.status).toBe('blocked');
    expect(frontmatter.title).toBe('Ship API');
    expect(frontmatter.owner).toBeUndefined();
    expect(frontmatter.labels).toEqual(['task', 'platform']);
    expect(frontmatter.metadata).toEqual({ date: '2026-02-14', by: 'alice' });
    expect(frontmatter.empty).toBeUndefined();
  });

  it('renders template documents and validates required/enum constraints', () => {
    const definition: PrimitiveTemplateDefinition = {
      name: 'task',
      primitive: 'task',
      format: 'schema',
      body: '# {{title}}\nAssigned to {{owner}}',
      fields: {
        status: { type: 'string', required: true, default: 'open', enum: ['open', 'blocked', 'done'] },
        owner: { type: 'string', required: true, default: '{{owner}}' },
      },
    };

    const rendered = renderDocumentFromTemplate(definition, {
      title: 'Ship parser',
      now: new Date('2026-02-14T06:07:08.000Z'),
      variables: { owner: 'alice' },
      overrides: { status: 'blocked' },
    });

    expect(rendered.frontmatter).toMatchObject({ status: 'blocked', owner: 'alice' });
    expect(rendered.content).toContain('# Ship parser');
    expect(rendered.content).toContain('Assigned to alice');
    expect(rendered.markdown).toContain('status: blocked');

    expect(validateFrontmatter(definition, { status: 'blocked' })).toEqual([
      {
        field: 'owner',
        kind: 'required',
        message: 'Required field "owner" is missing.',
      },
    ]);

    expect(validateFrontmatter(definition, { status: 'invalid', owner: 'alice' })).toEqual([
      {
        field: 'status',
        kind: 'enum',
        message: '"invalid" is not a valid value for "status". Expected one of: open, blocked, done.',
      },
    ]);
  });
});
