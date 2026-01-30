/**
 * ClawVault Search Engine - qmd Backend
 * Uses qmd CLI for BM25 and vector search
 */

import { execSync } from 'child_process';
import * as path from 'path';
import { Document, SearchResult, SearchOptions } from '../types.js';

/**
 * QMD search result format
 */
interface QmdResult {
  docid: string;
  score: number;
  file: string;
  title: string;
  snippet: string;
}

/**
 * Execute qmd command and return parsed JSON
 */
function execQmd(args: string[]): QmdResult[] {
  try {
    const result = execSync(`qmd ${args.join(' ')}`, {
      encoding: 'utf-8',
      stdio: ['pipe', 'pipe', 'pipe'],
      maxBuffer: 10 * 1024 * 1024 // 10MB
    });
    
    // Parse JSON output
    const parsed = JSON.parse(result.trim());
    return Array.isArray(parsed) ? parsed : [];
  } catch (err: any) {
    // Check if there's output in stderr or stdout
    if (err.stdout) {
      try {
        const parsed = JSON.parse(err.stdout.trim());
        return Array.isArray(parsed) ? parsed : [];
      } catch {
        // Not JSON
      }
    }
    console.error(`qmd error: ${err.message}`);
    return [];
  }
}

/**
 * Check if qmd is available
 */
export function hasQmd(): boolean {
  try {
    execSync('which qmd', { stdio: 'pipe' });
    return true;
  } catch {
    return false;
  }
}

/**
 * Trigger qmd update (reindex)
 */
export function qmdUpdate(): void {
  try {
    execSync('qmd update', { stdio: 'inherit' });
  } catch (err: any) {
    console.error(`qmd update failed: ${err.message}`);
  }
}

/**
 * Trigger qmd embed (create/update vector embeddings)
 */
export function qmdEmbed(): void {
  try {
    execSync('qmd embed', { stdio: 'inherit' });
  } catch (err: any) {
    console.error(`qmd embed failed: ${err.message}`);
  }
}

/**
 * QMD Search Engine - wraps qmd CLI
 */
export class SearchEngine {
  private documents: Map<string, Document> = new Map();
  private collection: string = 'clawvault';
  private vaultPath: string = '';

  /**
   * Set the collection name (usually vault name)
   */
  setCollection(name: string): void {
    this.collection = name;
  }

  /**
   * Set the vault path for file resolution
   */
  setVaultPath(vaultPath: string): void {
    this.vaultPath = vaultPath;
  }

  /**
   * Add or update a document in the local cache
   * Note: qmd indexing happens via qmd update command
   */
  addDocument(doc: Document): void {
    this.documents.set(doc.id, doc);
  }

  /**
   * Remove a document from the local cache
   */
  removeDocument(id: string): void {
    this.documents.delete(id);
  }

  /**
   * No-op for qmd - indexing is managed externally
   */
  rebuildIDF(): void {
    // qmd handles this
  }

  /**
   * BM25 search via qmd
   */
  search(query: string, options: SearchOptions = {}): SearchResult[] {
    const { 
      limit = 10, 
      minScore = 0,
      category, 
      tags,
      fullContent = false 
    } = options;
    
    if (!query.trim()) return [];

    // Build qmd command
    const args = [
      'search',
      `"${query.replace(/"/g, '\\"')}"`,
      '-n', String(limit * 2), // Request extra for filtering
      '--json'
    ];

    // Add collection filter if we have one
    if (this.collection) {
      args.push('-c', this.collection);
    }

    const qmdResults = execQmd(args);
    
    // Convert qmd results to ClawVault format
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
  vsearch(query: string, options: SearchOptions = {}): SearchResult[] {
    const { 
      limit = 10, 
      minScore = 0,
      category, 
      tags,
      fullContent = false 
    } = options;
    
    if (!query.trim()) return [];

    // Build qmd vsearch command
    const args = [
      'vsearch',
      `"${query.replace(/"/g, '\\"')}"`,
      '-n', String(limit * 2), // Request extra for filtering
      '--json'
    ];

    // Add collection filter if we have one
    if (this.collection) {
      args.push('-c', this.collection);
    }

    const qmdResults = execQmd(args);
    
    // Convert qmd results to ClawVault format
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
  query(query: string, options: SearchOptions = {}): SearchResult[] {
    const { 
      limit = 10, 
      minScore = 0,
      category, 
      tags,
      fullContent = false 
    } = options;
    
    if (!query.trim()) return [];

    // Build qmd query command (combined search with reranking)
    const args = [
      'query',
      `"${query.replace(/"/g, '\\"')}"`,
      '-n', String(limit * 2),
      '--json'
    ];

    if (this.collection) {
      args.push('-c', this.collection);
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
  private convertResults(
    qmdResults: QmdResult[], 
    options: SearchOptions
  ): SearchResult[] {
    const { limit = 10, minScore = 0, category, tags, fullContent = false } = options;
    
    const results: SearchResult[] = [];
    
    // Normalize scores - qmd uses different scales
    const maxScore = qmdResults[0]?.score || 1;
    
    for (const qr of qmdResults) {
      // Extract file path from qmd:// URI
      const filePath = this.qmdUriToPath(qr.file);
      const relativePath = this.vaultPath 
        ? path.relative(this.vaultPath, filePath)
        : filePath;
      
      // Get document from cache or create minimal one
      const docId = relativePath.replace(/\.md$/, '');
      let doc = this.documents.get(docId);
      
      // Determine category from path
      const parts = relativePath.split(path.sep);
      const docCategory = parts.length > 1 ? parts[0] : 'root';
      
      // Apply category filter
      if (category && docCategory !== category) continue;
      
      // Apply tag filter (only if we have the document cached)
      if (tags && tags.length > 0 && doc) {
        const docTags = new Set(doc.tags);
        if (!tags.some(t => docTags.has(t))) continue;
      }
      
      // Normalize score to 0-1 range
      const normalizedScore = maxScore > 0 ? qr.score / maxScore : 0;
      
      // Apply min score filter
      if (normalizedScore < minScore) continue;
      
      // Create document if not cached
      if (!doc) {
        doc = {
          id: docId,
          path: filePath,
          category: docCategory,
          title: qr.title || path.basename(relativePath, '.md'),
          content: fullContent ? '' : '', // Content loaded separately if needed
          frontmatter: {},
          links: [],
          tags: [],
          modified: new Date()
        };
      }
      
      results.push({
        document: fullContent ? doc : { ...doc, content: '' },
        score: normalizedScore,
        snippet: this.cleanSnippet(qr.snippet),
        matchedTerms: [] // qmd doesn't provide this
      });
      
      if (results.length >= limit) break;
    }
    
    return results;
  }

  /**
   * Convert qmd:// URI to file path
   */
  private qmdUriToPath(uri: string): string {
    // qmd://collection/path/to/file.md -> actual path
    if (uri.startsWith('qmd://')) {
      const withoutScheme = uri.slice(6); // Remove 'qmd://'
      const slashIndex = withoutScheme.indexOf('/');
      if (slashIndex > -1) {
        // Get collection name and relative path
        const collectionName = withoutScheme.slice(0, slashIndex);
        const relativePath = withoutScheme.slice(slashIndex + 1);
        
        // Try to resolve from vault path first
        if (this.vaultPath) {
          return path.join(this.vaultPath, relativePath);
        }
        
        // Fallback: try common paths
        const homeDir = process.env.HOME || '/home/frame';
        const possiblePaths = [
          path.join(homeDir, 'clawd/memory', relativePath),
          path.join(homeDir, 'clawd', collectionName, relativePath),
          relativePath
        ];
        
        for (const p of possiblePaths) {
          // Return first possibility (caller can verify existence)
          return p;
        }
      }
    }
    
    // Return as-is if not a qmd:// URI
    return uri;
  }

  /**
   * Clean up qmd snippet format
   */
  private cleanSnippet(snippet: string): string {
    if (!snippet) return '';
    
    // Remove diff-style markers like "@@ -2,4 @@ (1 before, 67 after)"
    return snippet
      .replace(/@@ [-+]?\d+,?\d* @@ \([^)]+\)/g, '')
      .trim()
      .split('\n')
      .slice(0, 3)
      .join('\n')
      .slice(0, 300);
  }

  /**
   * Get all cached documents
   */
  getAllDocuments(): Document[] {
    return [...this.documents.values()];
  }

  /**
   * Get document count
   */
  get size(): number {
    return this.documents.size;
  }

  /**
   * Clear the local document cache
   */
  clear(): void {
    this.documents.clear();
  }

  /**
   * Export documents for persistence
   */
  export(): { documents: Document[]; } {
    return {
      documents: [...this.documents.values()]
    };
  }

  /**
   * Import from persisted data
   */
  import(data: { documents: Document[]; }): void {
    this.clear();
    for (const doc of data.documents) {
      this.addDocument(doc);
    }
  }
}

/**
 * Find wiki-links in content
 */
export function extractWikiLinks(content: string): string[] {
  const matches = content.match(/\[\[([^\]]+)\]\]/g) || [];
  return matches.map(m => m.slice(2, -2).toLowerCase());
}

/**
 * Find tags in content (#tag format)
 */
export function extractTags(content: string): string[] {
  const matches = content.match(/#[\w-]+/g) || [];
  return [...new Set(matches.map(m => m.slice(1).toLowerCase()))];
}
