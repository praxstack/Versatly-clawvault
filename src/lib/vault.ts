/**
 * ClawVault - The elephant's memory
 */

import * as fs from 'fs';
import * as path from 'path';
import matter from 'gray-matter';
import { glob } from 'glob';
import {
  VaultConfig,
  VaultMeta,
  Document,
  SearchResult,
  SearchOptions,
  StoreOptions,
  SyncOptions,
  SyncResult,
  DEFAULT_CATEGORIES,
  Category
} from '../types.js';
import { SearchEngine, extractWikiLinks, extractTags, hasQmd, qmdUpdate, qmdEmbed } from './search.js';

const CONFIG_FILE = '.clawvault.json';
const INDEX_FILE = '.clawvault-index.json';

export class ClawVault {
  private config: VaultConfig;
  private search: SearchEngine;
  private initialized: boolean = false;

  constructor(vaultPath: string) {
    this.config = {
      path: path.resolve(vaultPath),
      name: path.basename(vaultPath),
      categories: DEFAULT_CATEGORIES
    };
    this.search = new SearchEngine();
    this.search.setVaultPath(this.config.path);
    this.search.setCollection(this.config.name);
  }

  /**
   * Initialize a new vault
   */
  async init(options: Partial<VaultConfig> = {}): Promise<void> {
    const vaultPath = this.config.path;
    
    // Merge options
    this.config = { ...this.config, ...options };
    
    // Create vault directory
    if (!fs.existsSync(vaultPath)) {
      fs.mkdirSync(vaultPath, { recursive: true });
    }

    // Create category directories
    for (const category of this.config.categories) {
      const catPath = path.join(vaultPath, category);
      if (!fs.existsSync(catPath)) {
        fs.mkdirSync(catPath, { recursive: true });
      }
    }

    // Create templates
    await this.createTemplates();

    // Create README
    const readmePath = path.join(vaultPath, 'README.md');
    if (!fs.existsSync(readmePath)) {
      fs.writeFileSync(readmePath, this.generateReadme());
    }

    // Save config
    const configPath = path.join(vaultPath, CONFIG_FILE);
    const meta: VaultMeta = {
      name: this.config.name,
      version: '1.0.0',
      created: new Date().toISOString(),
      lastUpdated: new Date().toISOString(),
      categories: this.config.categories,
      documentCount: 0
    };
    fs.writeFileSync(configPath, JSON.stringify(meta, null, 2));

    this.initialized = true;
  }

  /**
   * Load an existing vault
   */
  async load(): Promise<void> {
    const vaultPath = this.config.path;
    const configPath = path.join(vaultPath, CONFIG_FILE);

    if (!fs.existsSync(configPath)) {
      throw new Error(`Not a ClawVault: ${vaultPath} (missing ${CONFIG_FILE})`);
    }

    const meta: VaultMeta = JSON.parse(fs.readFileSync(configPath, 'utf-8'));
    this.config.name = meta.name;
    this.config.categories = meta.categories;

    // Configure search engine with vault info
    this.search.setVaultPath(this.config.path);
    // Use qmdCollection if set, otherwise fall back to vault name
    this.search.setCollection(meta.qmdCollection || this.config.name);

    // Index all documents (local cache)
    await this.reindex();
    this.initialized = true;
  }

  /**
   * Reindex all documents
   */
  async reindex(): Promise<number> {
    this.search.clear();
    
    const files = await glob('**/*.md', {
      cwd: this.config.path,
      ignore: ['**/node_modules/**', '**/.*']
    });

    for (const file of files) {
      const doc = await this.loadDocument(file);
      if (doc) {
        this.search.addDocument(doc);
      }
    }

    // Save index
    await this.saveIndex();

    return this.search.size;
  }

  /**
   * Load a document from disk
   */
  private async loadDocument(relativePath: string): Promise<Document | null> {
    try {
      const fullPath = path.join(this.config.path, relativePath);
      const content = fs.readFileSync(fullPath, 'utf-8');
      const { data: frontmatter, content: body } = matter(content);
      const stats = fs.statSync(fullPath);

      const parts = relativePath.split(path.sep);
      const category = parts.length > 1 ? parts[0] : 'root';
      const filename = path.basename(relativePath, '.md');

      return {
        id: relativePath.replace(/\.md$/, ''),
        path: fullPath,
        category,
        title: (frontmatter.title as string) || filename,
        content: body,
        frontmatter,
        links: extractWikiLinks(body),
        tags: extractTags(body),
        modified: stats.mtime
      };
    } catch (err) {
      console.error(`Error loading ${relativePath}:`, err);
      return null;
    }
  }

  /**
   * Store a new document
   */
  async store(options: StoreOptions): Promise<Document> {
    const { 
      category, 
      title, 
      content, 
      frontmatter = {}, 
      overwrite = false,
      qmdUpdate: triggerUpdate = false,
      qmdEmbed: triggerEmbed = false
    } = options;

    // Create filename from title
    const filename = this.slugify(title) + '.md';
    const relativePath = path.join(category, filename);
    const fullPath = path.join(this.config.path, relativePath);

    // Check if exists
    if (fs.existsSync(fullPath) && !overwrite) {
      throw new Error(`Document already exists: ${relativePath}. Use overwrite: true to replace.`);
    }

    // Ensure category directory exists
    const categoryPath = path.join(this.config.path, category);
    if (!fs.existsSync(categoryPath)) {
      fs.mkdirSync(categoryPath, { recursive: true });
    }

    // Build frontmatter with date
    const fm = {
      title,
      date: new Date().toISOString().split('T')[0],
      ...frontmatter
    };

    // Write file
    const fileContent = matter.stringify(content, fm);
    fs.writeFileSync(fullPath, fileContent);

    // Load and index the document
    const doc = await this.loadDocument(relativePath);
    if (doc) {
      this.search.addDocument(doc);
      await this.saveIndex();
    }

    // Trigger qmd reindex if requested
    if (triggerUpdate || triggerEmbed) {
      if (hasQmd()) {
        qmdUpdate();
        if (triggerEmbed) {
          qmdEmbed();
        }
      }
    }

    return doc!;
  }

  /**
   * Quick store to inbox
   */
  async capture(note: string, title?: string): Promise<Document> {
    const autoTitle = title || `note-${Date.now()}`;
    return this.store({
      category: 'inbox',
      title: autoTitle,
      content: note
    });
  }

  /**
   * Search the vault (BM25 via qmd)
   */
  async find(query: string, options: SearchOptions = {}): Promise<SearchResult[]> {
    return this.search.search(query, options);
  }

  /**
   * Semantic/vector search (via qmd vsearch)
   */
  async vsearch(query: string, options: SearchOptions = {}): Promise<SearchResult[]> {
    return this.search.vsearch(query, options);
  }

  /**
   * Combined search with query expansion (via qmd query)
   */
  async query(query: string, options: SearchOptions = {}): Promise<SearchResult[]> {
    return this.search.query(query, options);
  }

  /**
   * Get a document by ID or path
   */
  async get(idOrPath: string): Promise<Document | null> {
    // Normalize path
    const normalized = idOrPath.replace(/\.md$/, '');
    const docs = this.search.getAllDocuments();
    return docs.find(d => d.id === normalized) || null;
  }

  /**
   * List documents in a category
   */
  async list(category?: string): Promise<Document[]> {
    const docs = this.search.getAllDocuments();
    if (category) {
      return docs.filter(d => d.category === category);
    }
    return docs;
  }

  /**
   * Sync vault to another location (for Obsidian on Windows, etc.)
   */
  async sync(options: SyncOptions): Promise<SyncResult> {
    const { target, deleteOrphans = false, dryRun = false } = options;
    const result: SyncResult = {
      copied: [],
      deleted: [],
      unchanged: [],
      errors: []
    };

    // Get all source files
    const sourceFiles = await glob('**/*.md', {
      cwd: this.config.path,
      ignore: ['**/node_modules/**']
    });

    // Ensure target exists
    if (!dryRun && !fs.existsSync(target)) {
      fs.mkdirSync(target, { recursive: true });
    }

    // Copy files
    for (const file of sourceFiles) {
      const sourcePath = path.join(this.config.path, file);
      const targetPath = path.join(target, file);

      try {
        const sourceStats = fs.statSync(sourcePath);
        let shouldCopy = true;

        if (fs.existsSync(targetPath)) {
          const targetStats = fs.statSync(targetPath);
          if (sourceStats.mtime <= targetStats.mtime) {
            result.unchanged.push(file);
            shouldCopy = false;
          }
        }

        if (shouldCopy) {
          if (!dryRun) {
            const targetDir = path.dirname(targetPath);
            if (!fs.existsSync(targetDir)) {
              fs.mkdirSync(targetDir, { recursive: true });
            }
            fs.copyFileSync(sourcePath, targetPath);
          }
          result.copied.push(file);
        }
      } catch (err) {
        result.errors.push(`${file}: ${err}`);
      }
    }

    // Handle orphans in target
    if (deleteOrphans) {
      const targetFiles = await glob('**/*.md', { cwd: target });
      const sourceSet = new Set(sourceFiles);
      
      for (const file of targetFiles) {
        if (!sourceSet.has(file)) {
          if (!dryRun) {
            fs.unlinkSync(path.join(target, file));
          }
          result.deleted.push(file);
        }
      }
    }

    return result;
  }

  /**
   * Get vault statistics
   */
  async stats(): Promise<{
    documents: number;
    categories: { [key: string]: number };
    links: number;
    tags: string[];
  }> {
    const docs = this.search.getAllDocuments();
    const categories: { [key: string]: number } = {};
    const allTags = new Set<string>();
    let totalLinks = 0;

    for (const doc of docs) {
      categories[doc.category] = (categories[doc.category] || 0) + 1;
      totalLinks += doc.links.length;
      doc.tags.forEach(t => allTags.add(t));
    }

    return {
      documents: docs.length,
      categories,
      links: totalLinks,
      tags: [...allTags].sort()
    };
  }

  /**
   * Get all categories
   */
  getCategories(): Category[] {
    return this.config.categories;
  }

  /**
   * Check if vault is initialized
   */
  isInitialized(): boolean {
    return this.initialized;
  }

  /**
   * Get vault path
   */
  getPath(): string {
    return this.config.path;
  }

  /**
   * Get vault name
   */
  getName(): string {
    return this.config.name;
  }

  // === Private helpers ===

  private slugify(text: string): string {
    return text
      .toLowerCase()
      .replace(/[^\w\s-]/g, '')
      .replace(/\s+/g, '-')
      .replace(/-+/g, '-')
      .trim();
  }

  private async saveIndex(): Promise<void> {
    const indexPath = path.join(this.config.path, INDEX_FILE);
    const data = this.search.export();
    fs.writeFileSync(indexPath, JSON.stringify(data, null, 2));

    // Update config
    const configPath = path.join(this.config.path, CONFIG_FILE);
    if (fs.existsSync(configPath)) {
      const meta: VaultMeta = JSON.parse(fs.readFileSync(configPath, 'utf-8'));
      meta.lastUpdated = new Date().toISOString();
      meta.documentCount = this.search.size;
      fs.writeFileSync(configPath, JSON.stringify(meta, null, 2));
    }
  }

  private async createTemplates(): Promise<void> {
    const templatesPath = path.join(this.config.path, 'templates');
    
    const templates: { [key: string]: string } = {
      'decision.md': `---
title: "Decision: {{title}}"
date: {{date}}
status: pending
---

# Decision: {{title}}

## Context
What situation led to this decision?

## Options Considered
1. **Option A** — pros/cons
2. **Option B** — pros/cons

## Decision
What was decided?

## Reasoning
Why this choice?

## Outcome
[Fill in later] What happened as a result?

## Related
- [[people/]]
- [[projects/]]

#decision`,

      'pattern.md': `---
title: "Pattern: {{title}}"
date: {{date}}
confidence: medium
frequency: situational
---

# Pattern: {{title}}

## Description
What is the pattern?

## Evidence
- {{date}}: Example 1
- {{date}}: Example 2

## Implications
How should I act on this pattern?

## Related
- [[people/]]
- [[patterns/]]

#pattern`,

      'person.md': `---
title: "{{name}}"
date: {{date}}
role: ""
---

# {{name}}

**Role:** 
**First Mentioned:** {{date}}

## Context
How do we know this person?

## Key Facts
- 

## Interactions
- {{date}}: 

## Related
- [[people/]]
- [[projects/]]

#person`,

      'project.md': `---
title: "{{title}}"
date: {{date}}
status: active
---

# {{title}}

## Overview
What is this project?

## Goals
- 

## Progress
- {{date}}: Started

## People
- [[people/]]

## Decisions
- [[decisions/]]

#project`,

      'preference.md': `---
title: "Preference: {{title}}"
date: {{date}}
category: general
---

# Preference: {{title}}

## What
Description of the preference

## Why
Reasoning behind it

## Examples
- Example 1
- Example 2

#preference`
    };

    for (const [filename, content] of Object.entries(templates)) {
      const filePath = path.join(templatesPath, filename);
      if (!fs.existsSync(filePath)) {
        fs.writeFileSync(filePath, content);
      }
    }
  }

  private generateReadme(): string {
    return `# ${this.config.name} 🐘

An elephant never forgets.

## Structure

${this.config.categories.map(c => `- \`/${c}/\` — ${this.getCategoryDescription(c)}`).join('\n')}

## Quick Search

\`\`\`bash
clawvault search "query"
\`\`\`

## Quick Capture

\`\`\`bash
clawvault store --category inbox --title "note" --content "..."
\`\`\`

---

*Managed by [ClawVault](https://github.com/Versatly/clawvault)*
`;
  }

  private getCategoryDescription(category: string): string {
    const descriptions: { [key: string]: string } = {
      preferences: 'Likes, dislikes, and preferences',
      decisions: 'Choices with context and reasoning',
      patterns: 'Recurring behaviors observed',
      people: 'One file per person mentioned',
      projects: 'Active projects and ventures',
      goals: 'Long-term and short-term goals',
      transcripts: 'Session summaries',
      inbox: 'Quick capture → process later',
      templates: 'Templates for each document type'
    };
    return descriptions[category] || category;
  }
}

/**
 * Find and open the nearest vault (walks up directory tree)
 */
export async function findVault(startPath: string = process.cwd()): Promise<ClawVault | null> {
  let current = path.resolve(startPath);
  
  while (current !== path.dirname(current)) {
    const configPath = path.join(current, CONFIG_FILE);
    if (fs.existsSync(configPath)) {
      const vault = new ClawVault(current);
      await vault.load();
      return vault;
    }
    current = path.dirname(current);
  }
  
  return null;
}

/**
 * Create a new vault
 */
export async function createVault(vaultPath: string, options: Partial<VaultConfig> = {}): Promise<ClawVault> {
  const vault = new ClawVault(vaultPath);
  await vault.init(options);
  return vault;
}
