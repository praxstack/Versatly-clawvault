import type { Document, SearchOptions, SearchResult, VaultSearchConfig } from '../types.js';
import { EmbeddingStore } from './embedding-store.js';
import { cosineSimilarity, embedText, resolveEmbeddingConfig } from './hosted-embeddings.js';
import { crossEncoderRerank, resolveRerankConfig } from './hosted-rerank.js';

interface IndexedChunk {
  id: string;
  docId: string;
  text: string;
  termFreq: Map<string, number>;
  length: number;
}

interface CandidateResult {
  id: string;
  docId: string;
  snippet: string;
  score: number;
  matchedTerms: string[];
}

const BM25_K1 = 1.2;
const BM25_B = 0.75;
const RRF_K = 60;
const DEFAULT_CHUNK_SIZE = 700;
const DEFAULT_CHUNK_OVERLAP = 100;

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^\p{L}\p{N}_-]+/u)
    .map((token) => token.trim())
    .filter((token) => token.length > 1);
}

function normalizeScore(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return 0;
  if (max <= min) return value > 0 ? 1 : 0;
  return (value - min) / (max - min);
}

function normalizeCandidateScores(candidates: CandidateResult[]): CandidateResult[] {
  if (!candidates.length) return candidates;
  const values = candidates.map((item) => item.score).filter((value) => Number.isFinite(value));
  const min = Math.min(...values);
  const max = Math.max(...values);
  return candidates.map((candidate) => ({
    ...candidate,
    score: normalizeScore(candidate.score, min, max)
  }));
}

export class InProcessSearchEngine {
  private vaultPath = '';
  private config: VaultSearchConfig = {};
  private readonly documents = new Map<string, Document>();
  private readonly chunks = new Map<string, IndexedChunk>();
  private readonly chunkIdsByDoc = new Map<string, string[]>();
  private readonly documentFrequency = new Map<string, number>();
  private totalChunkLength = 0;
  private readonly embeddingStore = new EmbeddingStore(process.cwd());
  private embeddingStoreLoaded = false;

  setVaultPath(vaultPath: string): void {
    this.vaultPath = vaultPath;
    this.embeddingStore.setVaultPath(vaultPath);
    this.embeddingStoreLoaded = false;
  }

  setConfig(config?: VaultSearchConfig): void {
    this.config = config ?? {};
  }

  addDocument(doc: Document): void {
    const existing = this.documents.get(doc.id);
    if (existing) {
      this.removeDocument(existing.id);
    }
    this.documents.set(doc.id, doc);
    const chunkIds: string[] = [];
    const chunkTexts = this.splitIntoChunks(`${doc.title}\n${doc.content}`);

    for (let index = 0; index < chunkTexts.length; index += 1) {
      const text = chunkTexts[index];
      const terms = tokenize(text);
      if (!terms.length) continue;
      const termFreq = new Map<string, number>();
      for (const term of terms) {
        termFreq.set(term, (termFreq.get(term) ?? 0) + 1);
      }

      const chunkId = `${doc.id}#${index + 1}`;
      chunkIds.push(chunkId);
      this.totalChunkLength += terms.length;
      this.chunks.set(chunkId, {
        id: chunkId,
        docId: doc.id,
        text,
        termFreq,
        length: terms.length
      });

      for (const term of termFreq.keys()) {
        this.documentFrequency.set(term, (this.documentFrequency.get(term) ?? 0) + 1);
      }
    }

    this.chunkIdsByDoc.set(doc.id, chunkIds);
  }

  removeDocument(docId: string): void {
    const existingChunkIds = this.chunkIdsByDoc.get(docId) ?? [];
    for (const chunkId of existingChunkIds) {
      const chunk = this.chunks.get(chunkId);
      if (!chunk) continue;
      this.totalChunkLength = Math.max(0, this.totalChunkLength - chunk.length);
      for (const term of chunk.termFreq.keys()) {
        const next = (this.documentFrequency.get(term) ?? 0) - 1;
        if (next <= 0) {
          this.documentFrequency.delete(term);
        } else {
          this.documentFrequency.set(term, next);
        }
      }
      this.chunks.delete(chunkId);
    }
    this.chunkIdsByDoc.delete(docId);
    this.documents.delete(docId);
  }

  clear(): void {
    this.documents.clear();
    this.chunks.clear();
    this.chunkIdsByDoc.clear();
    this.documentFrequency.clear();
    this.totalChunkLength = 0;
  }

  getAllDocuments(): Document[] {
    return [...this.documents.values()];
  }

  get size(): number {
    return this.documents.size;
  }

  export(): { documents: Document[] } {
    return { documents: this.getAllDocuments() };
  }

  import(data: { documents: Document[] }): void {
    this.clear();
    for (const doc of data.documents) {
      this.addDocument(doc);
    }
  }

  async search(query: string, options: SearchOptions = {}): Promise<SearchResult[]> {
    if (!query.trim()) return [];
    const limit = Math.max(1, options.limit ?? 10);
    const bm25Candidates = this.runBm25(query, options, limit * 5);
    const semanticRanks = await this.getSemanticRanks(query, options);
    let fused = this.fuseHybrid(bm25Candidates, semanticRanks, limit * 5);
    fused = await this.applyCrossEncoderRerank(query, fused);
    return this.toSearchResults(fused, options, limit);
  }

  async vsearch(query: string, options: SearchOptions = {}): Promise<SearchResult[]> {
    if (!query.trim()) return [];
    const limit = Math.max(1, options.limit ?? 10);
    const semanticRanks = await this.getSemanticRanks(query, options);
    if (semanticRanks.length === 0) {
      return [];
    }

    const candidates: CandidateResult[] = [];
    for (const { docId, score } of semanticRanks) {
      const doc = this.documents.get(docId);
      if (!doc || !this.matchesFilters(doc, options)) continue;
      candidates.push({
        id: `${docId}#semantic`,
        docId,
        snippet: this.buildSnippet(doc.content, []),
        score,
        matchedTerms: []
      });
    }

    const reranked = await this.applyCrossEncoderRerank(query, candidates);
    return this.toSearchResults(reranked, options, limit);
  }

  async query(queryText: string, options: SearchOptions = {}): Promise<SearchResult[]> {
    return this.search(queryText, options);
  }

  private splitIntoChunks(content: string): string[] {
    const normalized = content.replace(/\r\n/g, '\n').trim();
    if (!normalized) return [];

    const chunkSize = Math.max(200, this.config.chunkSize ?? DEFAULT_CHUNK_SIZE);
    const overlap = Math.max(0, Math.min(chunkSize - 1, this.config.chunkOverlap ?? DEFAULT_CHUNK_OVERLAP));
    const chunks: string[] = [];

    let start = 0;
    while (start < normalized.length) {
      let end = Math.min(normalized.length, start + chunkSize);
      if (end < normalized.length) {
        const boundary = normalized.slice(end, Math.min(normalized.length, end + 100)).search(/\s/);
        if (boundary >= 0) {
          end += boundary;
        }
      }

      const piece = normalized.slice(start, end).trim();
      if (piece) {
        chunks.push(piece);
      }
      if (end >= normalized.length) break;
      start = Math.max(0, end - overlap);
    }

    return chunks;
  }

  private runBm25(query: string, options: SearchOptions, topK: number): CandidateResult[] {
    const queryTerms = tokenize(query);
    if (!queryTerms.length || this.chunks.size === 0) {
      return [];
    }
    const uniqueTerms = [...new Set(queryTerms)];
    const chunkCount = this.chunks.size;
    const avgChunkLength = chunkCount > 0 ? this.totalChunkLength / chunkCount : 1;
    const candidates: CandidateResult[] = [];

    for (const chunk of this.chunks.values()) {
      const doc = this.documents.get(chunk.docId);
      if (!doc || !this.matchesFilters(doc, options)) continue;
      let score = 0;
      const matchedTerms: string[] = [];

      for (const term of uniqueTerms) {
        const termFreq = chunk.termFreq.get(term) ?? 0;
        if (termFreq === 0) continue;
        matchedTerms.push(term);
        const docFreq = this.documentFrequency.get(term) ?? 0;
        const idf = Math.log((chunkCount - docFreq + 0.5) / (docFreq + 0.5) + 1);
        const tfNorm = (termFreq * (BM25_K1 + 1))
          / (termFreq + BM25_K1 * (1 - BM25_B + BM25_B * (chunk.length / Math.max(1, avgChunkLength))));
        score += idf * tfNorm;
      }

      if (score <= 0) continue;
      candidates.push({
        id: chunk.id,
        docId: chunk.docId,
        snippet: this.buildSnippet(chunk.text, matchedTerms),
        score,
        matchedTerms
      });
    }

    return normalizeCandidateScores(
      candidates.sort((left, right) => right.score - left.score).slice(0, topK)
    );
  }

  private async getSemanticRanks(
    query: string,
    options: SearchOptions
  ): Promise<Array<{ docId: string; score: number; rank: number }>> {
    const embeddingConfig = resolveEmbeddingConfig(this.config);
    if (!embeddingConfig) {
      return [];
    }
    if (!this.loadEmbeddingStoreIfNeeded(embeddingConfig.provider, embeddingConfig.model)) {
      return [];
    }

    let queryEmbedding: Float32Array;
    try {
      queryEmbedding = await embedText(query, embeddingConfig, { isQuery: true });
    } catch {
      return [];
    }

    const ranked = Array.from(this.documents.values())
      .filter((doc) => this.matchesFilters(doc, options))
      .map((doc) => {
        const vector = this.embeddingStore.get(doc.id);
        if (!vector) {
          return null;
        }
        return {
          docId: doc.id,
          score: cosineSimilarity(queryEmbedding, vector)
        };
      })
      .filter((entry): entry is { docId: string; score: number } => entry !== null)
      .sort((left, right) => right.score - left.score)
      .slice(0, Math.max(20, (options.limit ?? 10) * 5));

    if (ranked.length === 0) {
      return [];
    }

    const min = Math.min(...ranked.map((entry) => entry.score));
    const max = Math.max(...ranked.map((entry) => entry.score));
    return ranked.map((entry, index) => ({
      docId: entry.docId,
      rank: index,
      score: normalizeScore(entry.score, min, max)
    }));
  }

  private fuseHybrid(
    bm25Candidates: CandidateResult[],
    semanticRanks: Array<{ docId: string; score: number; rank: number }>,
    topK: number
  ): CandidateResult[] {
    if (bm25Candidates.length === 0) {
      return semanticRanks.map((entry) => {
        const doc = this.documents.get(entry.docId)!;
        return {
          id: `${entry.docId}#semantic`,
          docId: entry.docId,
          snippet: this.buildSnippet(doc.content, []),
          score: 1 / (RRF_K + entry.rank + 1),
          matchedTerms: []
        };
      }).slice(0, topK);
    }

    const semanticRankMap = new Map<string, { rank: number; score: number }>(
      semanticRanks.map((entry) => [entry.docId, { rank: entry.rank, score: entry.score }])
    );
    const fused: CandidateResult[] = [];

    for (let index = 0; index < bm25Candidates.length; index += 1) {
      const candidate = bm25Candidates[index];
      const bm25Rrf = 0.65 / (RRF_K + index + 1);
      const semantic = semanticRankMap.get(candidate.docId);
      const semanticRrf = semantic ? 0.35 / (RRF_K + semantic.rank + 1) : 0;
      fused.push({
        ...candidate,
        score: bm25Rrf + semanticRrf
      });
    }

    const seenDocs = new Set(fused.map((entry) => entry.docId));
    for (const entry of semanticRanks) {
      if (seenDocs.has(entry.docId)) continue;
      const doc = this.documents.get(entry.docId);
      if (!doc) continue;
      fused.push({
        id: `${entry.docId}#semantic`,
        docId: entry.docId,
        snippet: this.buildSnippet(doc.content, []),
        matchedTerms: [],
        score: 0.35 / (RRF_K + entry.rank + 1)
      });
    }

    return normalizeCandidateScores(
      fused.sort((left, right) => right.score - left.score).slice(0, topK)
    );
  }

  private async applyCrossEncoderRerank(query: string, candidates: CandidateResult[]): Promise<CandidateResult[]> {
    const rerankConfig = resolveRerankConfig(this.config);
    if (!rerankConfig || candidates.length === 0) {
      return candidates;
    }

    const texts = candidates.map((candidate) => {
      const doc = this.documents.get(candidate.docId);
      const title = doc?.title ?? candidate.docId;
      return `${title}\n${candidate.snippet}`.trim();
    });

    const rerankScores = await crossEncoderRerank(query, texts, rerankConfig);
    if (!rerankScores) {
      return candidates;
    }

    const normalizedRerank = normalizeCandidateScores(
      candidates.map((candidate, index) => ({
        ...candidate,
        score: rerankScores[index] ?? 0
      }))
    );

    const weighted = candidates.map((candidate, index) => ({
      ...candidate,
      score: ((1 - rerankConfig.weight) * candidate.score)
        + (rerankConfig.weight * normalizedRerank[index].score)
    }));

    return weighted.sort((left, right) => right.score - left.score);
  }

  private toSearchResults(candidates: CandidateResult[], options: SearchOptions, limit: number): SearchResult[] {
    const minScore = options.minScore ?? 0;
    const boosted = candidates
      .map((candidate) => {
        const doc = this.documents.get(candidate.docId);
        if (!doc) return null;
        const temporal = options.temporalBoost ? this.getRecencyFactor(doc.modified) : 1;
        return {
          candidate,
          doc,
          score: candidate.score * temporal
        };
      })
      .filter((entry): entry is { candidate: CandidateResult; doc: Document; score: number } => entry !== null)
      .filter((entry) => entry.score >= minScore)
      .sort((left, right) => right.score - left.score)
      .slice(0, limit);

    return boosted.map((entry) => ({
      document: options.fullContent ? entry.doc : { ...entry.doc, content: '' },
      score: entry.score,
      snippet: entry.candidate.snippet,
      matchedTerms: entry.candidate.matchedTerms
    }));
  }

  private matchesFilters(doc: Document, options: SearchOptions): boolean {
    if (options.category && doc.category !== options.category) {
      return false;
    }
    if (options.tags?.length) {
      const docTags = new Set(doc.tags.map((tag) => tag.toLowerCase()));
      const hasTag = options.tags.some((tag) => docTags.has(tag.toLowerCase()));
      if (!hasTag) return false;
    }
    return true;
  }

  private getRecencyFactor(modifiedAt: Date): number {
    const ageMs = Math.max(0, Date.now() - modifiedAt.getTime());
    const ageDays = ageMs / (24 * 60 * 60 * 1000);
    if (ageDays < 1) return 1;
    if (ageDays <= 7) return 0.9;
    return 0.7;
  }

  private buildSnippet(text: string, matchedTerms: string[]): string {
    const normalized = text.replace(/\s+/g, ' ').trim();
    if (!normalized) return '';
    if (!matchedTerms.length) {
      return normalized.slice(0, 260);
    }

    const lower = normalized.toLowerCase();
    const firstTerm = matchedTerms.find((term) => lower.includes(term.toLowerCase()));
    if (!firstTerm) {
      return normalized.slice(0, 260);
    }
    const start = Math.max(0, lower.indexOf(firstTerm.toLowerCase()) - 80);
    const end = Math.min(normalized.length, start + 260);
    return normalized.slice(start, end).trim();
  }

  private loadEmbeddingStoreIfNeeded(provider: string, model: string): boolean {
    if (!this.embeddingStoreLoaded) {
      this.embeddingStore.load();
      this.embeddingStoreLoaded = true;
    }
    if (!this.embeddingStore.isCompatible(provider, model)) {
      return false;
    }
    this.embeddingStore.setSignature(provider, model);
    return true;
  }
}
