import { afterEach, describe, expect, it, vi } from 'vitest';
import { Command } from 'commander';

const { runObserverBenchmarkMock, formatSummaryMock } = vi.hoisted(() => ({
  runObserverBenchmarkMock: vi.fn(),
  formatSummaryMock: vi.fn()
}));

vi.mock('../observer/benchmark/runner.js', () => ({
  runObserverBenchmark: runObserverBenchmarkMock
}));

vi.mock('../observer/benchmark/format.js', () => ({
  formatObserverBenchmarkSummary: formatSummaryMock
}));

import { benchmarkObserverCommand, registerBenchmarkCommand } from './benchmark.js';

afterEach(() => {
  vi.clearAllMocks();
});

describe('benchmarkObserverCommand', () => {
  it('prints text summary by default', async () => {
    runObserverBenchmarkMock.mockResolvedValue({
      generatedAt: '2026-03-11T10:00:00.000Z',
      fixturesDir: '/tmp/fixtures',
      provider: 'mock',
      fixtureCount: 1,
      model: undefined,
      aggregate: {
        precision: 0.5,
        noiseRatio: 0.5,
        recall: 0.5,
        typeAccuracy: 0.5,
        keywordPreservation: 0.5,
        overall: 0.5
      },
      fixtures: []
    });
    formatSummaryMock.mockReturnValue('summary output');
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined);

    await benchmarkObserverCommand({
      fixturesDir: 'testdata/observer-benchmark'
    });

    expect(runObserverBenchmarkMock).toHaveBeenCalledWith(
      expect.objectContaining({
        provider: 'mock'
      })
    );
    expect(logSpy).toHaveBeenCalledWith('summary output');
    logSpy.mockRestore();
  });

  it('prints json report when requested', async () => {
    runObserverBenchmarkMock.mockResolvedValue({
      generatedAt: '2026-03-11T10:00:00.000Z',
      fixturesDir: '/tmp/fixtures',
      provider: 'mock',
      fixtureCount: 0,
      model: 'gpt-test',
      aggregate: {
        precision: 1,
        noiseRatio: 0,
        recall: 1,
        typeAccuracy: 1,
        keywordPreservation: 1,
        overall: 1
      },
      fixtures: []
    });
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined);

    await benchmarkObserverCommand({
      fixturesDir: 'testdata/observer-benchmark',
      reportFormat: 'json',
      model: 'gpt-test'
    });

    expect(logSpy).toHaveBeenCalledTimes(1);
    const printed = logSpy.mock.calls[0]?.[0];
    expect(typeof printed).toBe('string');
    expect(String(printed)).toContain('"provider": "mock"');
    logSpy.mockRestore();
  });

  it('throws for invalid provider', async () => {
    await expect(benchmarkObserverCommand({
      fixturesDir: 'testdata/observer-benchmark',
      provider: 'invalid-provider'
    })).rejects.toThrow('Invalid --provider value');
  });
});

describe('registerBenchmarkCommand', () => {
  it('registers benchmark observer CLI shape', () => {
    const program = new Command();
    registerBenchmarkCommand(program);

    const benchmark = program.commands.find((command) => command.name() === 'benchmark');
    expect(benchmark).toBeDefined();

    const observer = benchmark?.commands.find((command) => command.name() === 'observer');
    expect(observer).toBeDefined();
    const flags = observer?.options.map((option) => option.flags) ?? [];
    expect(flags).toEqual(expect.arrayContaining([
      '--fixtures-dir <path>',
      '--provider <provider>',
      '--model <model>',
      '--report-format <format>'
    ]));
  });
});
