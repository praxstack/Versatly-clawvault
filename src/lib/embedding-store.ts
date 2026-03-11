import * as crypto from 'crypto';
import * as fs from 'fs';
import * as path from 'path';

interface EmbeddingEntry {
  hash: string;
  embedding: number[];
}

interface EmbeddingEnvelope {
  version: 2;
  provider: string;
  model: string;
  vectors: Record<string, EmbeddingEntry | number[]>;
}

const CACHE_DIR = '.clawvault';
const CACHE_FILE = 'embeddings.bin.json';
const LEGACY_BIN_FILE = 'embeddings.bin';

export class EmbeddingStore {
  private vaultPath: string;
  private provider = 'none';
  private model = '';
  private vectors = new Map<string, { hash: string; embedding: Float32Array }>();
  private dirty = false;

  constructor(vaultPath: string) {
    this.vaultPath = path.resolve(vaultPath);
  }

  setVaultPath(vaultPath: string): void {
    this.vaultPath = path.resolve(vaultPath);
  }

  setSignature(provider: string, model: string): void {
    if (this.provider !== provider || this.model !== model) {
      this.provider = provider;
      this.model = model;
      this.dirty = true;
    }
  }

  getSignature(): { provider: string; model: string } {
    return { provider: this.provider, model: this.model };
  }

  isCompatible(provider: string, model: string): boolean {
    if (this.provider === 'none' || !this.model) {
      return this.vectors.size === 0;
    }
    return this.provider === provider && this.model === model;
  }

  load(): void {
    this.vectors.clear();
    const cachePath = this.getCachePath();
    if (!fs.existsSync(cachePath)) {
      return;
    }

    try {
      const raw = JSON.parse(fs.readFileSync(cachePath, 'utf-8')) as EmbeddingEnvelope | Record<string, number[]>;
      if ('version' in raw && 'vectors' in raw) {
        this.provider = typeof raw.provider === 'string' ? raw.provider : 'none';
        this.model = typeof raw.model === 'string' ? raw.model : '';
        for (const [docId, value] of Object.entries(raw.vectors)) {
          if (Array.isArray(value)) {
            this.vectors.set(docId, {
              hash: '',
              embedding: new Float32Array(value)
            });
            continue;
          }
          if (!value || !Array.isArray(value.embedding)) {
            continue;
          }
          this.vectors.set(docId, {
            hash: typeof value.hash === 'string' ? value.hash : '',
            embedding: new Float32Array(value.embedding)
          });
        }
      } else {
        this.provider = 'none';
        this.model = '';
        for (const [docId, embedding] of Object.entries(raw)) {
          if (Array.isArray(embedding)) {
            this.vectors.set(docId, {
              hash: '',
              embedding: new Float32Array(embedding)
            });
          }
        }
      }
      this.dirty = false;
    } catch {
      this.provider = 'none';
      this.model = '';
      this.vectors.clear();
      this.dirty = false;
    }
  }

  save(): void {
    if (!this.dirty) return;
    const cacheDir = path.dirname(this.getCachePath());
    fs.mkdirSync(cacheDir, { recursive: true });

    const vectors: EmbeddingEnvelope['vectors'] = {};
    for (const [docId, entry] of this.vectors.entries()) {
      vectors[docId] = {
        hash: entry.hash,
        embedding: Array.from(entry.embedding)
      };
    }

    const payload: EmbeddingEnvelope = {
      version: 2,
      provider: this.provider,
      model: this.model,
      vectors
    };

    fs.writeFileSync(this.getCachePath(), JSON.stringify(payload));
    const legacyBinPath = path.join(cacheDir, LEGACY_BIN_FILE);
    if (!fs.existsSync(legacyBinPath)) {
      fs.writeFileSync(legacyBinPath, '');
    }
    this.dirty = false;
  }

  get(docId: string): Float32Array | undefined {
    return this.vectors.get(docId)?.embedding;
  }

  getHash(docId: string): string | undefined {
    return this.vectors.get(docId)?.hash;
  }

  set(docId: string, hash: string, embedding: Float32Array): void {
    this.vectors.set(docId, { hash, embedding });
    this.dirty = true;
  }

  has(docId: string): boolean {
    return this.vectors.has(docId);
  }

  delete(docId: string): void {
    if (this.vectors.delete(docId)) {
      this.dirty = true;
    }
  }

  prune(validDocIds: Set<string>): void {
    let removedAny = false;
    for (const key of this.vectors.keys()) {
      if (!validDocIds.has(key)) {
        this.vectors.delete(key);
        removedAny = true;
      }
    }
    if (removedAny) {
      this.dirty = true;
    }
  }

  entries(): IterableIterator<[string, Float32Array]> {
    const pairs = Array.from(this.vectors.entries()).map(([id, entry]) => [id, entry.embedding] as [string, Float32Array]);
    return pairs[Symbol.iterator]();
  }

  get size(): number {
    return this.vectors.size;
  }

  clear(): void {
    if (this.vectors.size === 0) return;
    this.vectors.clear();
    this.dirty = true;
  }

  private getCachePath(): string {
    return path.join(this.vaultPath, CACHE_DIR, CACHE_FILE);
  }
}

export function computeEmbeddingHash(text: string): string {
  return crypto.createHash('sha1').update(text).digest('hex');
}
