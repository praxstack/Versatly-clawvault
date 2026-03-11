import * as path from 'path';
import type { Command } from 'commander';
import { formatObserverBenchmarkSummary } from '../observer/benchmark/format.js';
import { runObserverBenchmark } from '../observer/benchmark/runner.js';
import type { ObserverBenchmarkProvider, ObserverBenchmarkReportFormat } from '../observer/benchmark/types.js';

const PROVIDERS: ObserverBenchmarkProvider[] = [
  'mock',
  'anthropic',
  'openai',
  'gemini',
  'xai',
  'openai-compatible',
  'ollama',
  'minimax',
  'zai'
];

function parseReportFormat(raw: string | undefined): ObserverBenchmarkReportFormat {
  if (!raw || raw === 'text') {
    return 'text';
  }
  if (raw === 'json') {
    return 'json';
  }
  throw new Error(`Invalid --report-format value: ${raw}`);
}

function parseProvider(raw: string | undefined): ObserverBenchmarkProvider {
  if (!raw) {
    return 'mock';
  }
  if (!PROVIDERS.includes(raw as ObserverBenchmarkProvider)) {
    throw new Error(`Invalid --provider value: ${raw}`);
  }
  return raw as ObserverBenchmarkProvider;
}

export interface BenchmarkObserverCommandOptions {
  fixturesDir?: string;
  provider?: string;
  model?: string;
  reportFormat?: string;
}

export async function benchmarkObserverCommand(
  options: BenchmarkObserverCommandOptions
): Promise<void> {
  const provider = parseProvider(options.provider);
  const reportFormat = parseReportFormat(options.reportFormat);
  const fixturesDir = path.resolve(options.fixturesDir ?? path.join(process.cwd(), 'testdata', 'observer-benchmark'));

  const report = await runObserverBenchmark({
    fixturesDir,
    provider,
    model: options.model
  });

  if (reportFormat === 'json') {
    console.log(JSON.stringify(report, null, 2));
    return;
  }

  console.log(formatObserverBenchmarkSummary(report));
}

export function registerBenchmarkCommand(program: Command): void {
  const benchmark = program
    .command('benchmark')
    .description('Run quality benchmark harnesses');

  benchmark
    .command('observer')
    .description('Evaluate observer output quality against annotated transcripts')
    .option('--fixtures-dir <path>', 'Fixture root directory (default: testdata/observer-benchmark)')
    .option('--provider <provider>', 'Compression provider (mock|anthropic|openai|gemini|xai|openai-compatible|ollama|minimax|zai)', 'mock')
    .option('--model <model>', 'Model override for live provider runs')
    .option('--report-format <format>', 'Report output format (json|text)', 'text')
    .action(async (rawOptions: {
      fixturesDir?: string;
      provider?: string;
      model?: string;
      reportFormat?: string;
    }) => {
      await benchmarkObserverCommand({
        fixturesDir: rawOptions.fixturesDir,
        provider: rawOptions.provider,
        model: rawOptions.model,
        reportFormat: rawOptions.reportFormat
      });
    });
}
