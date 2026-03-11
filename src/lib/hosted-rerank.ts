import type { RerankProvider, VaultSearchConfig } from '../types.js';

interface RerankResultItem {
  index: number;
  relevance_score?: number;
  score?: number;
}

interface RerankApiResponse {
  results?: RerankResultItem[];
  data?: RerankResultItem[];
}

export interface ResolvedRerankConfig {
  provider: Exclude<RerankProvider, 'none'>;
  endpoint: string;
  model: string;
  apiKey: string;
  weight: number;
}

const DEFAULT_ENDPOINTS: Record<Exclude<RerankProvider, 'none'>, string> = {
  jina: 'https://api.jina.ai/v1/rerank',
  voyage: 'https://api.voyageai.com/v1/rerank',
  siliconflow: 'https://api.siliconflow.cn/v1/rerank',
  pinecone: 'https://api.pinecone.io/rerank'
};

const DEFAULT_MODELS: Record<Exclude<RerankProvider, 'none'>, string> = {
  jina: 'jina-reranker-v2-base-multilingual',
  voyage: 'rerank-2',
  siliconflow: 'BAAI/bge-reranker-v2-m3',
  pinecone: 'bge-reranker-v2-m3'
};

function clampWeight(value: number): number {
  if (!Number.isFinite(value)) return 0.6;
  if (value < 0) return 0;
  if (value > 1) return 1;
  return value;
}

function resolveApiKey(provider: Exclude<RerankProvider, 'none'>, configured?: string): string | undefined {
  if (configured?.trim()) return configured.trim();
  const envKeyByProvider: Record<Exclude<RerankProvider, 'none'>, string[]> = {
    jina: ['JINA_API_KEY'],
    voyage: ['VOYAGE_API_KEY'],
    siliconflow: ['SILICONFLOW_API_KEY'],
    pinecone: ['PINECONE_API_KEY']
  };
  for (const key of envKeyByProvider[provider]) {
    const value = process.env[key]?.trim();
    if (value) return value;
  }
  return process.env.RERANK_API_KEY?.trim();
}

export function resolveRerankConfig(searchConfig?: VaultSearchConfig): ResolvedRerankConfig | null {
  const provider = searchConfig?.rerank?.provider ?? 'none';
  if (provider === 'none') {
    return null;
  }

  const apiKey = resolveApiKey(provider, searchConfig?.rerank?.apiKey);
  if (!apiKey) {
    return null;
  }

  const endpoint = (searchConfig?.rerank?.endpoint?.trim() || DEFAULT_ENDPOINTS[provider]).replace(/\/+$/, '');
  const model = searchConfig?.rerank?.model?.trim() || DEFAULT_MODELS[provider];
  const weight = clampWeight(searchConfig?.rerank?.weight ?? 0.6);

  return {
    provider,
    endpoint,
    model,
    apiKey,
    weight
  };
}

export async function crossEncoderRerank(
  query: string,
  documents: string[],
  config: ResolvedRerankConfig
): Promise<number[] | null> {
  if (!documents.length) return null;

  try {
    const response = await fetch(config.endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${config.apiKey}`
      },
      body: JSON.stringify({
        model: config.model,
        query,
        documents,
        top_n: documents.length
      }),
      signal: AbortSignal.timeout(15000)
    });

    if (!response.ok) {
      return null;
    }

    const payload = await response.json() as RerankApiResponse;
    const items = payload.results ?? payload.data;
    if (!Array.isArray(items)) {
      return null;
    }

    const scores = new Array<number>(documents.length).fill(0);
    for (const item of items) {
      const score = Number(item.relevance_score ?? item.score ?? 0);
      if (Number.isFinite(item.index) && item.index >= 0 && item.index < documents.length) {
        scores[item.index] = score;
      }
    }
    return scores;
  } catch {
    return null;
  }
}
