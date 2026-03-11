import type { EmbeddingProvider, VaultSearchConfig } from '../types.js';

export interface ResolvedEmbeddingConfig {
  provider: Exclude<EmbeddingProvider, 'none'>;
  model: string;
  baseUrl: string;
  apiKey?: string;
}

const DEFAULT_OPENAI_MODEL = 'text-embedding-3-small';
const DEFAULT_GEMINI_MODEL = 'text-embedding-004';
const DEFAULT_OLLAMA_MODEL = 'nomic-embed-text';

function normalizeBaseUrl(url: string): string {
  return url.replace(/\/+$/, '');
}

function toFloat32(values: unknown): Float32Array {
  if (!Array.isArray(values)) {
    throw new Error('Embedding response did not contain a numeric vector.');
  }
  return new Float32Array(values.map((value) => Number(value)));
}

function resolveApiKey(provider: Exclude<EmbeddingProvider, 'none'>, configuredApiKey?: string): string | undefined {
  if (configuredApiKey?.trim()) return configuredApiKey.trim();
  if (provider === 'openai') return process.env.OPENAI_API_KEY?.trim();
  if (provider === 'gemini') return process.env.GEMINI_API_KEY?.trim() || process.env.GOOGLE_API_KEY?.trim();
  return undefined;
}

export function resolveEmbeddingConfig(searchConfig?: VaultSearchConfig): ResolvedEmbeddingConfig | null {
  const provider = searchConfig?.embeddings?.provider ?? 'none';
  if (provider === 'none') {
    return null;
  }

  if (provider === 'openai') {
    return {
      provider,
      model: searchConfig?.embeddings?.model?.trim() || DEFAULT_OPENAI_MODEL,
      baseUrl: normalizeBaseUrl(searchConfig?.embeddings?.baseUrl?.trim() || 'https://api.openai.com/v1'),
      apiKey: resolveApiKey(provider, searchConfig?.embeddings?.apiKey)
    };
  }

  if (provider === 'gemini') {
    return {
      provider,
      model: searchConfig?.embeddings?.model?.trim() || DEFAULT_GEMINI_MODEL,
      baseUrl: normalizeBaseUrl(searchConfig?.embeddings?.baseUrl?.trim() || 'https://generativelanguage.googleapis.com/v1beta'),
      apiKey: resolveApiKey(provider, searchConfig?.embeddings?.apiKey)
    };
  }

  return {
    provider,
    model: searchConfig?.embeddings?.model?.trim() || DEFAULT_OLLAMA_MODEL,
    baseUrl: normalizeBaseUrl(searchConfig?.embeddings?.baseUrl?.trim() || 'http://127.0.0.1:11434')
  };
}

async function openAiEmbed(text: string, config: ResolvedEmbeddingConfig): Promise<Float32Array> {
  if (!config.apiKey) {
    throw new Error('OpenAI embeddings require an API key (search.embeddings.apiKey or OPENAI_API_KEY).');
  }
  const response = await fetch(`${config.baseUrl}/embeddings`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${config.apiKey}`
    },
    body: JSON.stringify({
      model: config.model,
      input: text
    }),
    signal: AbortSignal.timeout(20000)
  });
  if (!response.ok) {
    throw new Error(`OpenAI embedding request failed with status ${response.status}.`);
  }
  const payload = await response.json() as { data?: Array<{ embedding?: unknown }>; };
  return toFloat32(payload.data?.[0]?.embedding);
}

function geminiModelPath(model: string): string {
  return model.startsWith('models/') ? model : `models/${model}`;
}

async function geminiEmbed(text: string, config: ResolvedEmbeddingConfig, isQuery: boolean): Promise<Float32Array> {
  if (!config.apiKey) {
    throw new Error('Gemini embeddings require an API key (search.embeddings.apiKey, GEMINI_API_KEY, or GOOGLE_API_KEY).');
  }

  const response = await fetch(`${config.baseUrl}/${geminiModelPath(config.model)}:embedContent`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-goog-api-key': config.apiKey
    },
    body: JSON.stringify({
      content: {
        parts: [{ text }]
      },
      taskType: isQuery ? 'RETRIEVAL_QUERY' : 'RETRIEVAL_DOCUMENT'
    }),
    signal: AbortSignal.timeout(20000)
  });

  if (!response.ok) {
    throw new Error(`Gemini embedding request failed with status ${response.status}.`);
  }

  const payload = await response.json() as {
    embedding?: { values?: unknown };
    embeddings?: Array<{ values?: unknown }>;
  };
  const values = payload.embedding?.values ?? payload.embeddings?.[0]?.values;
  return toFloat32(values);
}

async function ollamaEmbed(text: string, config: ResolvedEmbeddingConfig): Promise<Float32Array> {
  const primary = await fetch(`${config.baseUrl}/api/embed`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: config.model,
      input: text
    }),
    signal: AbortSignal.timeout(20000)
  });

  if (primary.ok) {
    const payload = await primary.json() as { embeddings?: unknown[]; };
    return toFloat32(payload.embeddings?.[0]);
  }

  const fallback = await fetch(`${config.baseUrl}/api/embeddings`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: config.model,
      prompt: text
    }),
    signal: AbortSignal.timeout(20000)
  });

  if (!fallback.ok) {
    throw new Error(`Ollama embedding request failed with status ${fallback.status}.`);
  }

  const payload = await fallback.json() as { embedding?: unknown };
  return toFloat32(payload.embedding);
}

export async function embedText(
  text: string,
  config: ResolvedEmbeddingConfig,
  options: { isQuery?: boolean } = {}
): Promise<Float32Array> {
  const normalized = text.trim();
  if (!normalized) {
    throw new Error('Cannot embed empty text.');
  }
  const truncated = normalized.slice(0, 12000);

  if (config.provider === 'openai') {
    return openAiEmbed(truncated, config);
  }
  if (config.provider === 'gemini') {
    return geminiEmbed(truncated, config, options.isQuery ?? false);
  }
  return ollamaEmbed(truncated, config);
}

export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length || a.length === 0) {
    return 0;
  }
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let index = 0; index < a.length; index += 1) {
    const av = a[index];
    const bv = b[index];
    dot += av * bv;
    normA += av * av;
    normB += bv * bv;
  }
  if (normA === 0 || normB === 0) {
    return 0;
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}
