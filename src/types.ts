/**
 * ClawVault Types - The elephant's memory structure
 */

export interface VaultConfig {
  /** Root path of the vault */
  path: string;
  /** Name of the vault */
  name: string;
  /** Categories to create on init */
  categories: string[];
  /** Custom templates path (optional) */
  templatesPath?: string;
}

export interface VaultMeta {
  name: string;
  version: string;
  created: string;
  lastUpdated: string;
  categories: string[];
  documentCount: number;
}

export interface Document {
  /** Unique ID (relative path without extension) */
  id: string;
  /** Full file path */
  path: string;
  /** Category (folder name) */
  category: string;
  /** Document title */
  title: string;
  /** Raw content */
  content: string;
  /** Frontmatter metadata */
  frontmatter: Record<string, unknown>;
  /** Extracted wiki-links [[like-this]] */
  links: string[];
  /** Tags extracted from content */
  tags: string[];
  /** Last modified timestamp */
  modified: Date;
}

export interface SearchResult {
  /** Document that matched */
  document: Document;
  /** Relevance score (0-1) */
  score: number;
  /** Matching snippet */
  snippet: string;
  /** Which terms matched */
  matchedTerms: string[];
}

export interface SearchOptions {
  /** Max results to return */
  limit?: number;
  /** Minimum score threshold (0-1) */
  minScore?: number;
  /** Filter by category */
  category?: string;
  /** Filter by tags */
  tags?: string[];
  /** Include full content in results */
  fullContent?: boolean;
}

export interface StoreOptions {
  /** Category to store in */
  category: string;
  /** Document title (used for filename) */
  title: string;
  /** Content body */
  content: string;
  /** Frontmatter metadata */
  frontmatter?: Record<string, unknown>;
  /** Override existing file */
  overwrite?: boolean;
}

export interface SyncOptions {
  /** Target directory to sync to */
  target: string;
  /** Delete files in target not in source */
  deleteOrphans?: boolean;
  /** Dry run - don't actually sync */
  dryRun?: boolean;
}

export interface SyncResult {
  copied: string[];
  deleted: string[];
  unchanged: string[];
  errors: string[];
}

export type Category = 
  | 'preferences'
  | 'decisions'
  | 'patterns'
  | 'people'
  | 'projects'
  | 'goals'
  | 'transcripts'
  | 'inbox'
  | 'templates'
  | string;

export const DEFAULT_CATEGORIES: Category[] = [
  'preferences',
  'decisions',
  'patterns',
  'people',
  'projects',
  'goals',
  'transcripts',
  'inbox',
  'templates'
];

export const DEFAULT_CONFIG: Partial<VaultConfig> = {
  categories: DEFAULT_CATEGORIES
};
