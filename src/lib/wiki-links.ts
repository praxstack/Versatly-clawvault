import * as path from 'path';

function stripInlineCode(line: string): string {
  let result = '';
  let cursor = 0;

  while (cursor < line.length) {
    if (line[cursor] !== '`') {
      result += line[cursor];
      cursor += 1;
      continue;
    }

    let fenceLength = 1;
    while (cursor + fenceLength < line.length && line[cursor + fenceLength] === '`') {
      fenceLength += 1;
    }

    const fence = '`'.repeat(fenceLength);
    const closeIndex = line.indexOf(fence, cursor + fenceLength);
    if (closeIndex === -1) {
      result += fence;
      cursor += fenceLength;
      continue;
    }

    cursor = closeIndex + fenceLength;
  }

  return result;
}

function parseFence(line: string): { marker: '`' | '~'; length: number } | null {
  const match = line.match(/^[ \t]{0,3}(`{3,}|~{3,})/);
  if (!match) return null;
  const marker = match[1][0];
  if (marker !== '`' && marker !== '~') return null;
  return { marker, length: match[1].length };
}

function isFenceClose(line: string, fence: { marker: '`' | '~'; length: number }): boolean {
  const re = new RegExp(`^[ \\t]{0,3}${fence.marker}{${fence.length},}[ \\t]*$`);
  return re.test(line);
}

/**
 * Remove markdown code regions from content:
 * - Fenced code blocks (``` and ~~~)
 * - Indented code block lines
 * - Inline code spans (`code`)
 */
export function stripMarkdownCode(markdownContent: string): string {
  const lines = markdownContent.split('\n');
  const visibleLines: string[] = [];
  let activeFence: { marker: '`' | '~'; length: number } | null = null;

  for (const line of lines) {
    if (activeFence) {
      if (isFenceClose(line, activeFence)) {
        activeFence = null;
      }
      visibleLines.push('');
      continue;
    }

    const openingFence = parseFence(line);
    if (openingFence) {
      activeFence = openingFence;
      visibleLines.push('');
      continue;
    }

    if (/^(?: {4}|\t)/.test(line)) {
      visibleLines.push('');
      continue;
    }

    visibleLines.push(stripInlineCode(line));
  }

  return visibleLines.join('\n');
}

/**
 * Extract raw wiki-link contents (without surrounding [[ ]]) from markdown.
 * Ignores links inside markdown code regions.
 */
export function extractRawWikiLinks(markdownContent: string): string[] {
  const content = stripMarkdownCode(markdownContent);
  const links: string[] = [];
  let cursor = 0;

  while (cursor < content.length) {
    const start = content.indexOf('[[', cursor);
    if (start === -1) break;

    const end = content.indexOf(']]', start + 2);
    if (end === -1) break;

    const candidate = content.slice(start + 2, end).trim();
    if (candidate) {
      links.push(candidate);
    }
    cursor = end + 2;
  }

  return links;
}

/**
 * Normalize a wiki-link target into vault-path form.
 * - Strips alias (|display)
 * - Strips heading (#section)
 * - Converts path separators to forward slashes
 * - Trims path segments and optional .md extension
 */
export function normalizeWikiLinkTarget(rawTarget: string): string {
  let value = rawTarget.trim();
  if (!value) return '';

  if (value.startsWith('[[') && value.endsWith(']]')) {
    value = value.slice(2, -2).trim();
  }

  const pipeIndex = value.indexOf('|');
  if (pipeIndex >= 0) {
    value = value.slice(0, pipeIndex);
  }

  if (value.startsWith('#')) {
    return '';
  }

  const hashIndex = value.indexOf('#');
  if (hashIndex >= 0) {
    value = value.slice(0, hashIndex);
  }

  value = value
    .trim()
    .replace(/\\/g, '/')
    .replace(/^\/+/, '');
  if (!value) return '';

  value = value
    .split('/')
    .map((segment) => segment.trim())
    .join('/');

  const normalizedPath = path.posix.normalize(value);
  if (!normalizedPath || normalizedPath === '.') {
    return '';
  }

  value = normalizedPath;
  if (value.toLowerCase().endsWith('.md')) {
    value = value.slice(0, -3);
  }

  return value.trim();
}
