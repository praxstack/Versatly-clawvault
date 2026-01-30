"use strict";
var __create = Object.create;
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __getProtoOf = Object.getPrototypeOf;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toESM = (mod, isNodeMode, target) => (target = mod != null ? __create(__getProtoOf(mod)) : {}, __copyProps(
  // If the importer is in node compatibility mode or this is not an ESM
  // file that has been converted to a CommonJS file using a Babel-
  // compatible transform (i.e. "__esModule" has not been set), then set
  // "default" to the CommonJS "module.exports" for node compatibility.
  isNodeMode || !mod || !mod.__esModule ? __defProp(target, "default", { value: mod, enumerable: true }) : target,
  mod
));
var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

// src/index.ts
var index_exports = {};
__export(index_exports, {
  ClawVault: () => ClawVault,
  DEFAULT_CATEGORIES: () => DEFAULT_CATEGORIES,
  DEFAULT_CONFIG: () => DEFAULT_CONFIG,
  SearchEngine: () => SearchEngine,
  VERSION: () => VERSION,
  createVault: () => createVault,
  extractTags: () => extractTags,
  extractWikiLinks: () => extractWikiLinks,
  findVault: () => findVault,
  hasQmd: () => hasQmd,
  qmdEmbed: () => qmdEmbed,
  qmdUpdate: () => qmdUpdate
});
module.exports = __toCommonJS(index_exports);

// src/lib/vault.ts
var fs = __toESM(require("fs"), 1);
var path2 = __toESM(require("path"), 1);
var import_gray_matter = __toESM(require("gray-matter"), 1);
var import_glob = require("glob");

// src/types.ts
var DEFAULT_CATEGORIES = [
  "preferences",
  "decisions",
  "patterns",
  "people",
  "projects",
  "goals",
  "transcripts",
  "inbox",
  "templates"
];
var DEFAULT_CONFIG = {
  categories: DEFAULT_CATEGORIES
};

// src/lib/search.ts
var import_child_process = require("child_process");
var path = __toESM(require("path"), 1);
function execQmd(args) {
  try {
    const result = (0, import_child_process.execSync)(`qmd ${args.join(" ")}`, {
      encoding: "utf-8",
      stdio: ["pipe", "pipe", "pipe"],
      maxBuffer: 10 * 1024 * 1024
      // 10MB
    });
    const parsed = JSON.parse(result.trim());
    return Array.isArray(parsed) ? parsed : [];
  } catch (err) {
    if (err.stdout) {
      try {
        const parsed = JSON.parse(err.stdout.trim());
        return Array.isArray(parsed) ? parsed : [];
      } catch {
      }
    }
    console.error(`qmd error: ${err.message}`);
    return [];
  }
}
function hasQmd() {
  try {
    (0, import_child_process.execSync)("which qmd", { stdio: "pipe" });
    return true;
  } catch {
    return false;
  }
}
function qmdUpdate() {
  try {
    (0, import_child_process.execSync)("qmd update", { stdio: "inherit" });
  } catch (err) {
    console.error(`qmd update failed: ${err.message}`);
  }
}
function qmdEmbed() {
  try {
    (0, import_child_process.execSync)("qmd embed", { stdio: "inherit" });
  } catch (err) {
    console.error(`qmd embed failed: ${err.message}`);
  }
}
var SearchEngine = class {
  documents = /* @__PURE__ */ new Map();
  collection = "clawvault";
  vaultPath = "";
  /**
   * Set the collection name (usually vault name)
   */
  setCollection(name) {
    this.collection = name;
  }
  /**
   * Set the vault path for file resolution
   */
  setVaultPath(vaultPath) {
    this.vaultPath = vaultPath;
  }
  /**
   * Add or update a document in the local cache
   * Note: qmd indexing happens via qmd update command
   */
  addDocument(doc) {
    this.documents.set(doc.id, doc);
  }
  /**
   * Remove a document from the local cache
   */
  removeDocument(id) {
    this.documents.delete(id);
  }
  /**
   * No-op for qmd - indexing is managed externally
   */
  rebuildIDF() {
  }
  /**
   * BM25 search via qmd
   */
  search(query, options = {}) {
    const {
      limit = 10,
      minScore = 0,
      category,
      tags,
      fullContent = false
    } = options;
    if (!query.trim()) return [];
    const args = [
      "search",
      `"${query.replace(/"/g, '\\"')}"`,
      "-n",
      String(limit * 2),
      // Request extra for filtering
      "--json"
    ];
    if (this.collection) {
      args.push("-c", this.collection);
    }
    const qmdResults = execQmd(args);
    return this.convertResults(qmdResults, {
      limit,
      minScore,
      category,
      tags,
      fullContent
    });
  }
  /**
   * Vector/semantic search via qmd vsearch
   */
  vsearch(query, options = {}) {
    const {
      limit = 10,
      minScore = 0,
      category,
      tags,
      fullContent = false
    } = options;
    if (!query.trim()) return [];
    const args = [
      "vsearch",
      `"${query.replace(/"/g, '\\"')}"`,
      "-n",
      String(limit * 2),
      // Request extra for filtering
      "--json"
    ];
    if (this.collection) {
      args.push("-c", this.collection);
    }
    const qmdResults = execQmd(args);
    return this.convertResults(qmdResults, {
      limit,
      minScore,
      category,
      tags,
      fullContent
    });
  }
  /**
   * Combined search with query expansion (qmd query command)
   */
  query(query, options = {}) {
    const {
      limit = 10,
      minScore = 0,
      category,
      tags,
      fullContent = false
    } = options;
    if (!query.trim()) return [];
    const args = [
      "query",
      `"${query.replace(/"/g, '\\"')}"`,
      "-n",
      String(limit * 2),
      "--json"
    ];
    if (this.collection) {
      args.push("-c", this.collection);
    }
    const qmdResults = execQmd(args);
    return this.convertResults(qmdResults, {
      limit,
      minScore,
      category,
      tags,
      fullContent
    });
  }
  /**
   * Convert qmd results to ClawVault SearchResult format
   */
  convertResults(qmdResults, options) {
    const { limit = 10, minScore = 0, category, tags, fullContent = false } = options;
    const results = [];
    const maxScore = qmdResults[0]?.score || 1;
    for (const qr of qmdResults) {
      const filePath = this.qmdUriToPath(qr.file);
      const relativePath = this.vaultPath ? path.relative(this.vaultPath, filePath) : filePath;
      const docId = relativePath.replace(/\.md$/, "");
      let doc = this.documents.get(docId);
      const parts = relativePath.split(path.sep);
      const docCategory = parts.length > 1 ? parts[0] : "root";
      if (category && docCategory !== category) continue;
      if (tags && tags.length > 0 && doc) {
        const docTags = new Set(doc.tags);
        if (!tags.some((t) => docTags.has(t))) continue;
      }
      const normalizedScore = maxScore > 0 ? qr.score / maxScore : 0;
      if (normalizedScore < minScore) continue;
      if (!doc) {
        doc = {
          id: docId,
          path: filePath,
          category: docCategory,
          title: qr.title || path.basename(relativePath, ".md"),
          content: fullContent ? "" : "",
          // Content loaded separately if needed
          frontmatter: {},
          links: [],
          tags: [],
          modified: /* @__PURE__ */ new Date()
        };
      }
      results.push({
        document: fullContent ? doc : { ...doc, content: "" },
        score: normalizedScore,
        snippet: this.cleanSnippet(qr.snippet),
        matchedTerms: []
        // qmd doesn't provide this
      });
      if (results.length >= limit) break;
    }
    return results;
  }
  /**
   * Convert qmd:// URI to file path
   */
  qmdUriToPath(uri) {
    if (uri.startsWith("qmd://")) {
      const withoutScheme = uri.slice(6);
      const slashIndex = withoutScheme.indexOf("/");
      if (slashIndex > -1) {
        const collectionName = withoutScheme.slice(0, slashIndex);
        const relativePath = withoutScheme.slice(slashIndex + 1);
        if (this.vaultPath) {
          return path.join(this.vaultPath, relativePath);
        }
        const homeDir = process.env.HOME || "/home/frame";
        const possiblePaths = [
          path.join(homeDir, "clawd/memory", relativePath),
          path.join(homeDir, "clawd", collectionName, relativePath),
          relativePath
        ];
        for (const p of possiblePaths) {
          return p;
        }
      }
    }
    return uri;
  }
  /**
   * Clean up qmd snippet format
   */
  cleanSnippet(snippet) {
    if (!snippet) return "";
    return snippet.replace(/@@ [-+]?\d+,?\d* @@ \([^)]+\)/g, "").trim().split("\n").slice(0, 3).join("\n").slice(0, 300);
  }
  /**
   * Get all cached documents
   */
  getAllDocuments() {
    return [...this.documents.values()];
  }
  /**
   * Get document count
   */
  get size() {
    return this.documents.size;
  }
  /**
   * Clear the local document cache
   */
  clear() {
    this.documents.clear();
  }
  /**
   * Export documents for persistence
   */
  export() {
    return {
      documents: [...this.documents.values()]
    };
  }
  /**
   * Import from persisted data
   */
  import(data) {
    this.clear();
    for (const doc of data.documents) {
      this.addDocument(doc);
    }
  }
};
function extractWikiLinks(content) {
  const matches = content.match(/\[\[([^\]]+)\]\]/g) || [];
  return matches.map((m) => m.slice(2, -2).toLowerCase());
}
function extractTags(content) {
  const matches = content.match(/#[\w-]+/g) || [];
  return [...new Set(matches.map((m) => m.slice(1).toLowerCase()))];
}

// src/lib/vault.ts
var CONFIG_FILE = ".clawvault.json";
var INDEX_FILE = ".clawvault-index.json";
var ClawVault = class {
  config;
  search;
  initialized = false;
  constructor(vaultPath) {
    this.config = {
      path: path2.resolve(vaultPath),
      name: path2.basename(vaultPath),
      categories: DEFAULT_CATEGORIES
    };
    this.search = new SearchEngine();
    this.search.setVaultPath(this.config.path);
    this.search.setCollection(this.config.name);
  }
  /**
   * Initialize a new vault
   */
  async init(options = {}) {
    const vaultPath = this.config.path;
    this.config = { ...this.config, ...options };
    if (!fs.existsSync(vaultPath)) {
      fs.mkdirSync(vaultPath, { recursive: true });
    }
    for (const category of this.config.categories) {
      const catPath = path2.join(vaultPath, category);
      if (!fs.existsSync(catPath)) {
        fs.mkdirSync(catPath, { recursive: true });
      }
    }
    await this.createTemplates();
    const readmePath = path2.join(vaultPath, "README.md");
    if (!fs.existsSync(readmePath)) {
      fs.writeFileSync(readmePath, this.generateReadme());
    }
    const configPath = path2.join(vaultPath, CONFIG_FILE);
    const meta = {
      name: this.config.name,
      version: "1.0.0",
      created: (/* @__PURE__ */ new Date()).toISOString(),
      lastUpdated: (/* @__PURE__ */ new Date()).toISOString(),
      categories: this.config.categories,
      documentCount: 0
    };
    fs.writeFileSync(configPath, JSON.stringify(meta, null, 2));
    this.initialized = true;
  }
  /**
   * Load an existing vault
   */
  async load() {
    const vaultPath = this.config.path;
    const configPath = path2.join(vaultPath, CONFIG_FILE);
    if (!fs.existsSync(configPath)) {
      throw new Error(`Not a ClawVault: ${vaultPath} (missing ${CONFIG_FILE})`);
    }
    const meta = JSON.parse(fs.readFileSync(configPath, "utf-8"));
    this.config.name = meta.name;
    this.config.categories = meta.categories;
    this.search.setVaultPath(this.config.path);
    this.search.setCollection(meta.qmdCollection || this.config.name);
    await this.reindex();
    this.initialized = true;
  }
  /**
   * Reindex all documents
   */
  async reindex() {
    this.search.clear();
    const files = await (0, import_glob.glob)("**/*.md", {
      cwd: this.config.path,
      ignore: ["**/node_modules/**", "**/.*"]
    });
    for (const file of files) {
      const doc = await this.loadDocument(file);
      if (doc) {
        this.search.addDocument(doc);
      }
    }
    await this.saveIndex();
    return this.search.size;
  }
  /**
   * Load a document from disk
   */
  async loadDocument(relativePath) {
    try {
      const fullPath = path2.join(this.config.path, relativePath);
      const content = fs.readFileSync(fullPath, "utf-8");
      const { data: frontmatter, content: body } = (0, import_gray_matter.default)(content);
      const stats = fs.statSync(fullPath);
      const parts = relativePath.split(path2.sep);
      const category = parts.length > 1 ? parts[0] : "root";
      const filename = path2.basename(relativePath, ".md");
      return {
        id: relativePath.replace(/\.md$/, ""),
        path: fullPath,
        category,
        title: frontmatter.title || filename,
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
  async store(options) {
    const {
      category,
      title,
      content,
      frontmatter = {},
      overwrite = false,
      qmdUpdate: triggerUpdate = false,
      qmdEmbed: triggerEmbed = false
    } = options;
    const filename = this.slugify(title) + ".md";
    const relativePath = path2.join(category, filename);
    const fullPath = path2.join(this.config.path, relativePath);
    if (fs.existsSync(fullPath) && !overwrite) {
      throw new Error(`Document already exists: ${relativePath}. Use overwrite: true to replace.`);
    }
    const categoryPath = path2.join(this.config.path, category);
    if (!fs.existsSync(categoryPath)) {
      fs.mkdirSync(categoryPath, { recursive: true });
    }
    const fm = {
      title,
      date: (/* @__PURE__ */ new Date()).toISOString().split("T")[0],
      ...frontmatter
    };
    const fileContent = import_gray_matter.default.stringify(content, fm);
    fs.writeFileSync(fullPath, fileContent);
    const doc = await this.loadDocument(relativePath);
    if (doc) {
      this.search.addDocument(doc);
      await this.saveIndex();
    }
    if (triggerUpdate || triggerEmbed) {
      if (hasQmd()) {
        qmdUpdate();
        if (triggerEmbed) {
          qmdEmbed();
        }
      }
    }
    return doc;
  }
  /**
   * Quick store to inbox
   */
  async capture(note, title) {
    const autoTitle = title || `note-${Date.now()}`;
    return this.store({
      category: "inbox",
      title: autoTitle,
      content: note
    });
  }
  /**
   * Search the vault (BM25 via qmd)
   */
  async find(query, options = {}) {
    return this.search.search(query, options);
  }
  /**
   * Semantic/vector search (via qmd vsearch)
   */
  async vsearch(query, options = {}) {
    return this.search.vsearch(query, options);
  }
  /**
   * Combined search with query expansion (via qmd query)
   */
  async query(query, options = {}) {
    return this.search.query(query, options);
  }
  /**
   * Get a document by ID or path
   */
  async get(idOrPath) {
    const normalized = idOrPath.replace(/\.md$/, "");
    const docs = this.search.getAllDocuments();
    return docs.find((d) => d.id === normalized) || null;
  }
  /**
   * List documents in a category
   */
  async list(category) {
    const docs = this.search.getAllDocuments();
    if (category) {
      return docs.filter((d) => d.category === category);
    }
    return docs;
  }
  /**
   * Sync vault to another location (for Obsidian on Windows, etc.)
   */
  async sync(options) {
    const { target, deleteOrphans = false, dryRun = false } = options;
    const result = {
      copied: [],
      deleted: [],
      unchanged: [],
      errors: []
    };
    const sourceFiles = await (0, import_glob.glob)("**/*.md", {
      cwd: this.config.path,
      ignore: ["**/node_modules/**"]
    });
    if (!dryRun && !fs.existsSync(target)) {
      fs.mkdirSync(target, { recursive: true });
    }
    for (const file of sourceFiles) {
      const sourcePath = path2.join(this.config.path, file);
      const targetPath = path2.join(target, file);
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
            const targetDir = path2.dirname(targetPath);
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
    if (deleteOrphans) {
      const targetFiles = await (0, import_glob.glob)("**/*.md", { cwd: target });
      const sourceSet = new Set(sourceFiles);
      for (const file of targetFiles) {
        if (!sourceSet.has(file)) {
          if (!dryRun) {
            fs.unlinkSync(path2.join(target, file));
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
  async stats() {
    const docs = this.search.getAllDocuments();
    const categories = {};
    const allTags = /* @__PURE__ */ new Set();
    let totalLinks = 0;
    for (const doc of docs) {
      categories[doc.category] = (categories[doc.category] || 0) + 1;
      totalLinks += doc.links.length;
      doc.tags.forEach((t) => allTags.add(t));
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
  getCategories() {
    return this.config.categories;
  }
  /**
   * Check if vault is initialized
   */
  isInitialized() {
    return this.initialized;
  }
  /**
   * Get vault path
   */
  getPath() {
    return this.config.path;
  }
  /**
   * Get vault name
   */
  getName() {
    return this.config.name;
  }
  // === Private helpers ===
  slugify(text) {
    return text.toLowerCase().replace(/[^\w\s-]/g, "").replace(/\s+/g, "-").replace(/-+/g, "-").trim();
  }
  async saveIndex() {
    const indexPath = path2.join(this.config.path, INDEX_FILE);
    const data = this.search.export();
    fs.writeFileSync(indexPath, JSON.stringify(data, null, 2));
    const configPath = path2.join(this.config.path, CONFIG_FILE);
    if (fs.existsSync(configPath)) {
      const meta = JSON.parse(fs.readFileSync(configPath, "utf-8"));
      meta.lastUpdated = (/* @__PURE__ */ new Date()).toISOString();
      meta.documentCount = this.search.size;
      fs.writeFileSync(configPath, JSON.stringify(meta, null, 2));
    }
  }
  async createTemplates() {
    const templatesPath = path2.join(this.config.path, "templates");
    const templates = {
      "decision.md": `---
title: "Decision: {{title}}"
date: {{date}}
status: pending
---

# Decision: {{title}}

## Context
What situation led to this decision?

## Options Considered
1. **Option A** \u2014 pros/cons
2. **Option B** \u2014 pros/cons

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
      "pattern.md": `---
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
      "person.md": `---
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
      "project.md": `---
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
      "preference.md": `---
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
      const filePath = path2.join(templatesPath, filename);
      if (!fs.existsSync(filePath)) {
        fs.writeFileSync(filePath, content);
      }
    }
  }
  generateReadme() {
    return `# ${this.config.name} \u{1F418}

An elephant never forgets.

## Structure

${this.config.categories.map((c) => `- \`/${c}/\` \u2014 ${this.getCategoryDescription(c)}`).join("\n")}

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
  getCategoryDescription(category) {
    const descriptions = {
      preferences: "Likes, dislikes, and preferences",
      decisions: "Choices with context and reasoning",
      patterns: "Recurring behaviors observed",
      people: "One file per person mentioned",
      projects: "Active projects and ventures",
      goals: "Long-term and short-term goals",
      transcripts: "Session summaries",
      inbox: "Quick capture \u2192 process later",
      templates: "Templates for each document type"
    };
    return descriptions[category] || category;
  }
};
async function findVault(startPath = process.cwd()) {
  let current = path2.resolve(startPath);
  while (current !== path2.dirname(current)) {
    const configPath = path2.join(current, CONFIG_FILE);
    if (fs.existsSync(configPath)) {
      const vault = new ClawVault(current);
      await vault.load();
      return vault;
    }
    current = path2.dirname(current);
  }
  return null;
}
async function createVault(vaultPath, options = {}) {
  const vault = new ClawVault(vaultPath);
  await vault.init(options);
  return vault;
}

// src/index.ts
var VERSION = "1.0.0";
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  ClawVault,
  DEFAULT_CATEGORIES,
  DEFAULT_CONFIG,
  SearchEngine,
  VERSION,
  createVault,
  extractTags,
  extractWikiLinks,
  findVault,
  hasQmd,
  qmdEmbed,
  qmdUpdate
});
