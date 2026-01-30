/**
 * ClawVault Search Engine
 * Simple but effective TF-IDF based semantic search
 */

import { Document, SearchResult, SearchOptions } from '../types.js';

interface TermFrequency {
  [term: string]: number;
}

interface DocumentIndex {
  id: string;
  terms: TermFrequency;
  termCount: number;
}

/**
 * Simple stemmer - reduces words to their root form
 */
function stem(word: string): string {
  word = word.toLowerCase();
  
  // Common suffix removal
  const suffixes = [
    'ingly', 'edly', 'tion', 'sion', 'ness', 'ment', 'able', 'ible',
    'ally', 'ful', 'less', 'ous', 'ive', 'ing', 'ed', 'ly', 's'
  ];
  
  for (const suffix of suffixes) {
    if (word.length > suffix.length + 2 && word.endsWith(suffix)) {
      return word.slice(0, -suffix.length);
    }
  }
  
  return word;
}

/**
 * Tokenize and normalize text
 */
function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/\[\[([^\]]+)\]\]/g, '$1') // Extract wiki-links
    .replace(/[^\w\s-]/g, ' ')
    .split(/\s+/)
    .filter(t => t.length > 2)
    .map(stem);
}

/**
 * Calculate term frequency for a document
 */
function calculateTF(tokens: string[]): TermFrequency {
  const tf: TermFrequency = {};
  for (const token of tokens) {
    tf[token] = (tf[token] || 0) + 1;
  }
  // Normalize by document length
  const len = tokens.length || 1;
  for (const term in tf) {
    tf[term] = tf[term] / len;
  }
  return tf;
}

/**
 * BM25 Search Engine - industry standard for text search
 */
export class SearchEngine {
  private documents: Map<string, Document> = new Map();
  private index: Map<string, DocumentIndex> = new Map();
  private idf: Map<string, number> = new Map();
  private avgDocLength: number = 0;
  
  // BM25 parameters
  private k1: number = 1.5;
  private b: number = 0.75;

  /**
   * Add or update a document in the index
   */
  addDocument(doc: Document): void {
    this.documents.set(doc.id, doc);
    
    // Combine title and content for indexing
    const text = `${doc.title} ${doc.title} ${doc.content}`; // Title weighted 2x
    const tokens = tokenize(text);
    
    this.index.set(doc.id, {
      id: doc.id,
      terms: calculateTF(tokens),
      termCount: tokens.length
    });
    
    this.rebuildIDF();
  }

  /**
   * Remove a document from the index
   */
  removeDocument(id: string): void {
    this.documents.delete(id);
    this.index.delete(id);
    this.rebuildIDF();
  }

  /**
   * Rebuild IDF scores (call after bulk updates)
   */
  rebuildIDF(): void {
    const docCount = this.index.size || 1;
    const termDocCounts: Map<string, number> = new Map();
    let totalLength = 0;
    
    // Count documents containing each term
    for (const [, idx] of this.index) {
      totalLength += idx.termCount;
      for (const term in idx.terms) {
        termDocCounts.set(term, (termDocCounts.get(term) || 0) + 1);
      }
    }
    
    this.avgDocLength = totalLength / docCount;
    
    // Calculate IDF for each term
    this.idf.clear();
    for (const [term, count] of termDocCounts) {
      // BM25 IDF formula
      this.idf.set(term, Math.log((docCount - count + 0.5) / (count + 0.5) + 1));
    }
  }

  /**
   * Search for documents matching query
   */
  search(query: string, options: SearchOptions = {}): SearchResult[] {
    const { 
      limit = 10, 
      minScore = 0.01, 
      category, 
      tags,
      fullContent = false 
    } = options;
    
    const queryTokens = tokenize(query);
    if (queryTokens.length === 0) return [];
    
    const scores: Map<string, number> = new Map();
    const matchedTerms: Map<string, Set<string>> = new Map();
    
    // Calculate BM25 score for each document
    for (const [docId, idx] of this.index) {
      const doc = this.documents.get(docId);
      if (!doc) continue;
      
      // Apply filters
      if (category && doc.category !== category) continue;
      if (tags && tags.length > 0) {
        const docTags = new Set(doc.tags);
        if (!tags.some(t => docTags.has(t))) continue;
      }
      
      let score = 0;
      const matched = new Set<string>();
      
      for (const term of queryTokens) {
        const tf = idx.terms[term] || 0;
        if (tf === 0) continue;
        
        const idf = this.idf.get(term) || 0;
        const docLen = idx.termCount;
        
        // BM25 scoring formula
        const numerator = tf * (this.k1 + 1);
        const denominator = tf + this.k1 * (1 - this.b + this.b * (docLen / this.avgDocLength));
        score += idf * (numerator / denominator);
        matched.add(term);
      }
      
      if (score > 0) {
        scores.set(docId, score);
        matchedTerms.set(docId, matched);
      }
    }
    
    // Sort by score and apply limit
    const sortedIds = [...scores.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, limit);
    
    // Normalize scores to 0-1 range
    const maxScore = sortedIds[0]?.[1] || 1;
    
    const results: SearchResult[] = [];
    for (const [docId, score] of sortedIds) {
      const normalizedScore = score / maxScore;
      if (normalizedScore < minScore) continue;
      
      const doc = this.documents.get(docId)!;
      const matched = matchedTerms.get(docId) || new Set();
      
      results.push({
        document: fullContent ? doc : {
          ...doc,
          content: '' // Omit content unless requested
        },
        score: normalizedScore,
        snippet: this.extractSnippet(doc.content, queryTokens),
        matchedTerms: [...matched]
      });
    }
    
    return results;
  }

  /**
   * Extract relevant snippet from content
   */
  private extractSnippet(content: string, queryTokens: string[]): string {
    const lines = content.split('\n');
    const querySet = new Set(queryTokens);
    
    // Find the line with most query term matches
    let bestLine = 0;
    let bestScore = 0;
    
    for (let i = 0; i < lines.length; i++) {
      const lineTokens = tokenize(lines[i]);
      const matches = lineTokens.filter(t => querySet.has(t)).length;
      if (matches > bestScore) {
        bestScore = matches;
        bestLine = i;
      }
    }
    
    // Extract context around best line
    const start = Math.max(0, bestLine - 1);
    const end = Math.min(lines.length, bestLine + 3);
    const snippet = lines.slice(start, end).join('\n').trim();
    
    // Truncate if too long
    if (snippet.length > 300) {
      return snippet.slice(0, 297) + '...';
    }
    
    return snippet || lines.slice(0, 3).join('\n').trim();
  }

  /**
   * Get all documents
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
   * Clear the index
   */
  clear(): void {
    this.documents.clear();
    this.index.clear();
    this.idf.clear();
    this.avgDocLength = 0;
  }

  /**
   * Export index for persistence
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
