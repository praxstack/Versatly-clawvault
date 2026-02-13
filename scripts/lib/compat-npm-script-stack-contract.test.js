import { describe, expect, it } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import {
  REQUIRED_COMPAT_ARTIFACT_PRODUCER_CONSUMER_CONTRACTS,
  REQUIRED_COMPAT_CI_REACHABLE_SCRIPT_NAMES,
  REQUIRED_COMPAT_CI_SEQUENCE,
  REQUIRED_COMPAT_ARTIFACT_CLI_DRIFT_PATHS,
  REQUIRED_COMPAT_ARTIFACT_STACK_SEQUENCE,
  REQUIRED_COMPAT_NPM_SCRIPT_NAMES,
  REQUIRED_COMPAT_REPORT_STACK_SEQUENCE,
  REQUIRED_COMPAT_SCRIPT_STACK_CONTRACT_TEST_PATHS,
  REQUIRED_COMPAT_SCRIPT_REFERENCE_SOURCES,
  REQUIRED_COMPAT_SUMMARY_STACK_SEQUENCE,
  REQUIRED_COMPAT_VALIDATOR_STACK_SEQUENCE
} from './compat-npm-script-contracts.mjs';
import {
  buildReachableNpmRunGraph,
  collectNpmRunTargets,
  hasReachableNpmRunCycle
} from './compat-npm-script-graph-utils.mjs';

function loadPackageScripts() {
  const packageJsonPath = path.resolve(process.cwd(), 'package.json');
  const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf-8'));
  return packageJson.scripts ?? {};
}

function expectContainsInOrder(value, parts, label) {
  let cursor = -1;
  for (const part of parts) {
    const nextIndex = value.indexOf(part, cursor + 1);
    expect(nextIndex, `${label} is missing segment: ${part}`).toBeGreaterThanOrEqual(0);
    cursor = nextIndex;
  }
}

function expectContainsExactlyOnce(value, parts, label) {
  for (const part of parts) {
    const firstIndex = value.indexOf(part);
    expect(firstIndex, `${label} contains no occurrence for segment: ${part}`).toBeGreaterThanOrEqual(0);
    expect(
      firstIndex,
      `${label} contains duplicate segment occurrence: ${part}`
    ).toBe(value.lastIndexOf(part));
  }
}

function expectContainsExactlyOnceInOrder(value, parts, label) {
  expectContainsInOrder(value, parts, label);
  expectContainsExactlyOnce(value, parts, label);
}

describe('compat npm script stack contracts', () => {
  it('keeps required compat script names present', () => {
    const scripts = loadPackageScripts();
    for (const scriptName of REQUIRED_COMPAT_NPM_SCRIPT_NAMES) {
      expect(typeof scripts[scriptName], `missing required script: ${scriptName}`).toBe('string');
    }
  });

  it('keeps npm-run references resolvable across required stack sources', () => {
    const scripts = loadPackageScripts();
    const { unresolvedScripts } = buildReachableNpmRunGraph({
      scripts,
      sourceScripts: REQUIRED_COMPAT_SCRIPT_REFERENCE_SOURCES
    });
    expect(
      [...unresolvedScripts],
      'missing npm-run targets detected in reachable compat stack graph'
    ).toEqual([]);
  });

  it('keeps reachable compat npm-run graph acyclic', () => {
    const scripts = loadPackageScripts();
    const { unresolvedScripts, adjacencyByScript } = buildReachableNpmRunGraph({
      scripts,
      sourceScripts: REQUIRED_COMPAT_SCRIPT_REFERENCE_SOURCES
    });
    expect(
      [...unresolvedScripts],
      'cannot evaluate cycle contracts with unresolved npm-run targets'
    ).toEqual([]);
    expect(
      hasReachableNpmRunCycle(adjacencyByScript, adjacencyByScript.keys()),
      'cycle detected in reachable compat npm-run graph'
    ).toBe(false);
  });

  it('keeps dedicated artifact CLI drift script wired to both validator suites', () => {
    const scripts = loadPackageScripts();
    const cliDriftScript = scripts['test:compat-artifact-cli-drift:fast'];
    expect(typeof cliDriftScript).toBe('string');
    expectContainsExactlyOnce(
      cliDriftScript,
      REQUIRED_COMPAT_ARTIFACT_CLI_DRIFT_PATHS,
      'test:compat-artifact-cli-drift:fast'
    );
  });

  it('keeps script-stack contract runner wired to all governance suites', () => {
    const scripts = loadPackageScripts();
    const stackContractScript = scripts['test:compat-script-stack-contract:fast'];
    expect(typeof stackContractScript).toBe('string');
    expectContainsExactlyOnce(
      stackContractScript,
      REQUIRED_COMPAT_SCRIPT_STACK_CONTRACT_TEST_PATHS,
      'test:compat-script-stack-contract:fast'
    );
  });

  it('keeps fast artifact stack ordering aligned with required contract gates', () => {
    const scripts = loadPackageScripts();
    const artifactStackScript = scripts['test:compat-artifact-stack:fast'];
    expect(typeof artifactStackScript).toBe('string');
    expectContainsExactlyOnceInOrder(
      artifactStackScript,
      REQUIRED_COMPAT_ARTIFACT_STACK_SEQUENCE,
      'test:compat-artifact-stack:fast'
    );
  });

  it('keeps fast report stack chained through validator/artifact stacks', () => {
    const scripts = loadPackageScripts();
    const reportStackScript = scripts['test:compat-report-stack:fast'];
    expect(typeof reportStackScript).toBe('string');
    expectContainsExactlyOnceInOrder(
      reportStackScript,
      REQUIRED_COMPAT_REPORT_STACK_SEQUENCE,
      'test:compat-report-stack:fast'
    );
  });

  it('keeps fast validator stack sequence ordered for verifier/schema gates', () => {
    const scripts = loadPackageScripts();
    const validatorStackScript = scripts['test:compat-validator-stack:fast'];
    expect(typeof validatorStackScript).toBe('string');
    expectContainsExactlyOnceInOrder(
      validatorStackScript,
      REQUIRED_COMPAT_VALIDATOR_STACK_SEQUENCE,
      'test:compat-validator-stack:fast'
    );
  });

  it('keeps fast summary stack chained through report stack', () => {
    const scripts = loadPackageScripts();
    const summaryFastScript = scripts['test:compat-summary:fast'];
    expect(typeof summaryFastScript).toBe('string');
    expectContainsExactlyOnceInOrder(
      summaryFastScript,
      REQUIRED_COMPAT_SUMMARY_STACK_SEQUENCE,
      'test:compat-summary:fast'
    );
  });

  it('keeps ci script ordered through core + compat gates', () => {
    const scripts = loadPackageScripts();
    const ciScript = scripts.ci;
    expect(typeof ciScript).toBe('string');
    expectContainsExactlyOnceInOrder(
      ciScript,
      REQUIRED_COMPAT_CI_SEQUENCE,
      'ci'
    );
  });

  it('keeps artifact producers ordered before their consumer gates', () => {
    const scripts = loadPackageScripts();
    for (const {
      scriptName,
      artifactFile,
      producerSegment,
      consumerSegment
    } of REQUIRED_COMPAT_ARTIFACT_PRODUCER_CONSUMER_CONTRACTS) {
      const command = scripts[scriptName];
      expect(typeof command, `missing script for producer/consumer contract: ${scriptName}`).toBe('string');
      const producerIndex = command.indexOf(producerSegment);
      const consumerIndex = command.indexOf(consumerSegment);
      expect(producerIndex, `${scriptName} missing producer segment for ${artifactFile}: ${producerSegment}`).toBeGreaterThanOrEqual(0);
      expect(consumerIndex, `${scriptName} missing consumer segment for ${artifactFile}: ${consumerSegment}`).toBeGreaterThanOrEqual(0);
      expect(
        producerIndex,
        `${scriptName} runs consumer before producer for ${artifactFile}`
      ).toBeLessThan(consumerIndex);
      expect(
        producerIndex,
        `${scriptName} contains duplicate producer segment for ${artifactFile}`
      ).toBe(command.lastIndexOf(producerSegment));
      expect(
        consumerIndex,
        `${scriptName} contains duplicate consumer segment for ${artifactFile}`
      ).toBe(command.lastIndexOf(consumerSegment));
    }
  });

  it('keeps required compat scripts reachable from ci npm-run graph', () => {
    const scripts = loadPackageScripts();
    const { unresolvedScripts, visitedScripts } = buildReachableNpmRunGraph({
      scripts,
      sourceScripts: ['ci']
    });
    expect(
      [...unresolvedScripts],
      'cannot evaluate ci reachability contracts with unresolved npm-run targets'
    ).toEqual([]);
    for (const requiredScriptName of REQUIRED_COMPAT_CI_REACHABLE_SCRIPT_NAMES) {
      expect(
        visitedScripts.has(requiredScriptName),
        `ci npm-run graph does not reach required compat script: ${requiredScriptName}`
      ).toBe(true);
    }
  });
});
