import type { ObserverBenchmarkReport } from './types.js';

function percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export function formatObserverBenchmarkSummary(report: ObserverBenchmarkReport): string {
  const lines: string[] = [];
  lines.push('Observer benchmark report');
  lines.push(`Generated: ${report.generatedAt}`);
  lines.push(`Fixtures: ${report.fixtureCount}`);
  lines.push(`Provider: ${report.provider}${report.model ? ` (${report.model})` : ''}`);
  lines.push('');
  lines.push('Aggregate metrics');
  lines.push(`- Precision: ${percent(report.aggregate.precision)} (noise ${percent(report.aggregate.noiseRatio)})`);
  lines.push(`- Recall (important events): ${percent(report.aggregate.recall)}`);
  lines.push(`- Keyword preservation: ${percent(report.aggregate.keywordPreservation)}`);
  lines.push(`- Type accuracy: ${percent(report.aggregate.typeAccuracy)}`);
  lines.push(`- Overall score: ${percent(report.aggregate.overall)}`);
  lines.push('');
  lines.push('Per-fixture');

  for (const fixture of report.fixtures) {
    lines.push(`- ${fixture.fixtureId}`);
    lines.push(`  overall=${percent(fixture.scoring.metrics.overall)} precision=${percent(fixture.scoring.metrics.precision)} recall=${percent(fixture.scoring.metrics.recall)} keyword=${percent(fixture.scoring.metrics.keywordPreservation)} type=${percent(fixture.scoring.metrics.typeAccuracy)}`);
    lines.push(`  matches=${fixture.scoring.matches.length} expected=${fixture.expectedCount} actual=${fixture.actualCount} missed-important=${fixture.scoring.missedImportant.length} noise=${fixture.scoring.noiseObservations.length}`);
    if (fixture.scoring.missingKeywords.length > 0) {
      lines.push(`  missing keywords: ${fixture.scoring.missingKeywords.slice(0, 5).join(', ')}${fixture.scoring.missingKeywords.length > 5 ? ' …' : ''}`);
    }
  }

  return lines.join('\n');
}
