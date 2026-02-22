import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { resolveOpenClawProvider, resolveLlmProvider, requestLlmCompletion } from './llm-provider.js';

describe('OpenClaw provider integration', () => {
  let tmpDir: string;
  let origHome: string | undefined;

  beforeEach(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'clawvault-oc-'));
    origHome = process.env.OPENCLAW_HOME;
    process.env.OPENCLAW_HOME = tmpDir;
  });

  afterEach(() => {
    if (origHome !== undefined) process.env.OPENCLAW_HOME = origHome;
    else delete process.env.OPENCLAW_HOME;
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  function writeModelsJson(providers: Record<string, unknown>) {
    const dir = path.join(tmpDir, 'agents', 'main', 'agent');
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, 'models.json'), JSON.stringify({ providers }));
  }

  describe('resolveOpenClawProvider', () => {
    it('returns null when no models.json exists', () => {
      expect(resolveOpenClawProvider()).toBeNull();
    });

    it('returns null when providers have no apiKey', () => {
      writeModelsJson({ test: { baseUrl: 'http://localhost:8080/v1', models: [{ id: 'gpt-4o' }] } });
      expect(resolveOpenClawProvider()).toBeNull();
    });

    it('resolves first provider with baseUrl and apiKey', () => {
      writeModelsJson({
        myProxy: {
          baseUrl: 'http://proxy.local:8317/v1/',
          apiKey: 'sk-test-123',
          api: 'openai-completions',
          models: [{ id: 'claude-opus-4-6', name: 'Claude Opus' }]
        }
      });
      const result = resolveOpenClawProvider();
      expect(result).not.toBeNull();
      expect(result!.baseUrl).toBe('http://proxy.local:8317/v1');
      expect(result!.apiKey).toBe('sk-test-123');
      expect(result!.defaultModel).toBe('claude-opus-4-6');
    });
  });

  describe('resolveLlmProvider', () => {
    it('returns openclaw when config is available', () => {
      writeModelsJson({
        p: { baseUrl: 'http://x/v1', apiKey: 'k', models: [{ id: 'm' }] }
      });
      // Clear other env keys
      const saved = { a: process.env.ANTHROPIC_API_KEY, o: process.env.OPENAI_API_KEY, g: process.env.GEMINI_API_KEY };
      delete process.env.ANTHROPIC_API_KEY;
      delete process.env.OPENAI_API_KEY;
      delete process.env.GEMINI_API_KEY;
      try {
        expect(resolveLlmProvider()).toBe('openclaw');
      } finally {
        if (saved.a) process.env.ANTHROPIC_API_KEY = saved.a;
        if (saved.o) process.env.OPENAI_API_KEY = saved.o;
        if (saved.g) process.env.GEMINI_API_KEY = saved.g;
      }
    });
  });

  describe('requestLlmCompletion with openclaw', () => {
    it('calls OpenAI-compatible endpoint and returns content', async () => {
      writeModelsJson({
        p: { baseUrl: 'http://mock/v1', apiKey: 'test-key', models: [{ id: 'test-model' }] }
      });

      const fetchImpl: typeof fetch = async (input, init) => {
        const url = typeof input === 'string' ? input : (input as Request).url;
        expect(url).toBe('http://mock/v1/chat/completions');
        const body = JSON.parse(init?.body as string);
        expect(body.model).toBe('test-model');
        expect(body.messages).toHaveLength(2); // system + user
        return {
          ok: true,
          json: async () => ({
            choices: [{ message: { content: 'test response' } }]
          })
        } as Response;
      };

      const result = await requestLlmCompletion({
        prompt: 'hello',
        systemPrompt: 'you are helpful',
        provider: 'openclaw',
        fetchImpl
      });
      expect(result).toBe('test response');
    });
  });
});
