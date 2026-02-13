import { describe, expect, it } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import {
  REQUIRED_COMPAT_CI_CHECKOUT_STEP_NAME,
  REQUIRED_COMPAT_CI_CHECKOUT_USES,
  REQUIRED_COMPAT_CI_FAILURE_UPLOAD_ARTIFACT_NAME,
  REQUIRED_COMPAT_CI_FAILURE_UPLOAD_CONDITION,
  REQUIRED_COMPAT_CI_FAILURE_UPLOAD_IF_NO_FILES_FOUND,
  REQUIRED_COMPAT_CI_FAILURE_UPLOAD_PATH,
  REQUIRED_COMPAT_CI_FAILURE_UPLOAD_STEP_NAME,
  REQUIRED_COMPAT_CI_INSTALL_COMMAND,
  REQUIRED_COMPAT_CI_INSTALL_STEP_NAME,
  REQUIRED_COMPAT_CI_PRIMARY_RUN_COMMAND,
  REQUIRED_COMPAT_CI_REPORT_DIR_ENV_KEY,
  REQUIRED_COMPAT_CI_REPORT_DIR_ENV_VALUE,
  REQUIRED_COMPAT_CI_PRIMARY_RUN_STEP_NAME,
  REQUIRED_COMPAT_CI_SETUP_NODE_CACHE,
  REQUIRED_COMPAT_CI_SETUP_NODE_STEP_NAME,
  REQUIRED_COMPAT_CI_SETUP_NODE_USES,
  REQUIRED_COMPAT_CI_SETUP_NODE_VERSION,
  REQUIRED_COMPAT_CI_STEP_SEQUENCE,
  REQUIRED_COMPAT_CI_UPLOAD_ARTIFACT_NAME,
  REQUIRED_COMPAT_CI_UPLOAD_ARTIFACT_FILES,
  REQUIRED_COMPAT_CI_UPLOAD_IF_NO_FILES_FOUND,
  REQUIRED_COMPAT_CI_UPLOAD_ARTIFACT_PATH_PREFIX,
  REQUIRED_COMPAT_CI_UPLOAD_STEP_NAME
} from './compat-npm-script-contracts.mjs';

function escapeRegex(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function loadCiWorkflowYaml() {
  const workflowPath = path.resolve(process.cwd(), '.github', 'workflows', 'ci.yml');
  return fs.readFileSync(workflowPath, 'utf-8');
}

function extractStepMetadata(workflowYaml, stepName) {
  const stepHeaderPattern = new RegExp(`\\n\\s+- name:\\s+${escapeRegex(stepName)}\\s*\\n`);
  const headerMatch = stepHeaderPattern.exec(workflowYaml);
  expect(headerMatch, `missing CI workflow step: ${stepName}`).toBeTruthy();
  const startIndex = headerMatch.index;
  const nextStepIndex = workflowYaml.indexOf('\n      - name:', startIndex + headerMatch[0].length);
  if (nextStepIndex < 0) {
    return {
      startIndex,
      block: workflowYaml.slice(startIndex)
    };
  }
  return {
    startIndex,
    block: workflowYaml.slice(startIndex, nextStepIndex)
  };
}

function extractStepBlock(workflowYaml, stepName) {
  return extractStepMetadata(workflowYaml, stepName).block;
}

function extractRunCommand(stepBlock, stepName) {
  const runLine = stepBlock
    .split('\n')
    .map((line) => line.trim())
    .find((line) => line.startsWith('run:'));
  expect(runLine, `step "${stepName}" must include a run command`).toBeTruthy();
  return runLine.replace(/^run:\s*/, '').trim();
}

function extractScalarField(stepBlock, fieldName, stepName) {
  const line = stepBlock
    .split('\n')
    .map((entry) => entry.trim())
    .find((entry) => entry.startsWith(`${fieldName}:`));
  expect(line, `step "${stepName}" must include ${fieldName}`).toBeTruthy();
  return line.replace(new RegExp(`^${escapeRegex(fieldName)}:\\s*`), '').trim();
}

function extractEnvField(stepBlock, envKey, stepName) {
  const line = stepBlock
    .split('\n')
    .map((entry) => entry.trim())
    .find((entry) => entry.startsWith(`${envKey}:`));
  expect(line, `step "${stepName}" must define env ${envKey}`).toBeTruthy();
  return line.replace(new RegExp(`^${escapeRegex(envKey)}:\\s*`), '').trim();
}

function extractUsesField(stepBlock, stepName) {
  return extractScalarField(stepBlock, 'uses', stepName);
}

function extractUploadArtifactPaths(stepBlock, stepName) {
  const lines = stepBlock.split('\n').map((line) => line.trim());
  const pathLineIndex = lines.findIndex((line) => line === 'path:' || line === 'path: |');
  expect(pathLineIndex, `step "${stepName}" must include a multiline path block`).toBeGreaterThanOrEqual(0);
  const pathLines = [];
  for (let index = pathLineIndex + 1; index < lines.length; index += 1) {
    const line = lines[index];
    if (!line) {
      continue;
    }
    if (line.startsWith('if-no-files-found:')) {
      break;
    }
    if (line.includes(':') && !line.startsWith('${{')) {
      break;
    }
    pathLines.push(line);
  }
  return pathLines;
}

describe('compat ci workflow contract', () => {
  it('keeps core CI step sequence ordered', () => {
    const workflowYaml = loadCiWorkflowYaml();
    let previousStepStartIndex = -1;
    for (const stepName of REQUIRED_COMPAT_CI_STEP_SEQUENCE) {
      const { startIndex } = extractStepMetadata(workflowYaml, stepName);
      expect(
        startIndex,
        `step "${stepName}" appears before previous required CI step in sequence`
      ).toBeGreaterThan(previousStepStartIndex);
      previousStepStartIndex = startIndex;
    }
  });

  it('keeps checkout/setup/install steps aligned with canonical CI environment contracts', () => {
    const workflowYaml = loadCiWorkflowYaml();
    const checkoutStepBlock = extractStepBlock(workflowYaml, REQUIRED_COMPAT_CI_CHECKOUT_STEP_NAME);
    const setupNodeStepBlock = extractStepBlock(workflowYaml, REQUIRED_COMPAT_CI_SETUP_NODE_STEP_NAME);
    const installStepBlock = extractStepBlock(workflowYaml, REQUIRED_COMPAT_CI_INSTALL_STEP_NAME);
    expect(extractUsesField(checkoutStepBlock, REQUIRED_COMPAT_CI_CHECKOUT_STEP_NAME)).toBe(REQUIRED_COMPAT_CI_CHECKOUT_USES);
    expect(extractUsesField(setupNodeStepBlock, REQUIRED_COMPAT_CI_SETUP_NODE_STEP_NAME)).toBe(REQUIRED_COMPAT_CI_SETUP_NODE_USES);
    expect(extractScalarField(setupNodeStepBlock, 'node-version', REQUIRED_COMPAT_CI_SETUP_NODE_STEP_NAME)).toBe(
      REQUIRED_COMPAT_CI_SETUP_NODE_VERSION
    );
    expect(extractScalarField(setupNodeStepBlock, 'cache', REQUIRED_COMPAT_CI_SETUP_NODE_STEP_NAME)).toBe(
      REQUIRED_COMPAT_CI_SETUP_NODE_CACHE
    );
    expect(extractRunCommand(installStepBlock, REQUIRED_COMPAT_CI_INSTALL_STEP_NAME)).toBe(REQUIRED_COMPAT_CI_INSTALL_COMMAND);
  });

  it('runs canonical ci command from primary run step', () => {
    const workflowYaml = loadCiWorkflowYaml();
    const stepBlock = extractStepBlock(workflowYaml, REQUIRED_COMPAT_CI_PRIMARY_RUN_STEP_NAME);
    const runCommand = extractRunCommand(stepBlock, REQUIRED_COMPAT_CI_PRIMARY_RUN_STEP_NAME);
    const reportDirValue = extractEnvField(
      stepBlock,
      REQUIRED_COMPAT_CI_REPORT_DIR_ENV_KEY,
      REQUIRED_COMPAT_CI_PRIMARY_RUN_STEP_NAME
    );
    expect(runCommand).toBe(REQUIRED_COMPAT_CI_PRIMARY_RUN_COMMAND);
    expect(reportDirValue).toBe(REQUIRED_COMPAT_CI_REPORT_DIR_ENV_VALUE);
  });

  it('uploads required compatibility artifact files in canonical order', () => {
    const workflowYaml = loadCiWorkflowYaml();
    const stepBlock = extractStepBlock(workflowYaml, REQUIRED_COMPAT_CI_UPLOAD_STEP_NAME);
    const artifactName = extractScalarField(stepBlock, 'name', REQUIRED_COMPAT_CI_UPLOAD_STEP_NAME);
    const ifNoFilesFoundValue = extractScalarField(stepBlock, 'if-no-files-found', REQUIRED_COMPAT_CI_UPLOAD_STEP_NAME);
    const uploadPaths = extractUploadArtifactPaths(stepBlock, REQUIRED_COMPAT_CI_UPLOAD_STEP_NAME);
    const expectedPaths = REQUIRED_COMPAT_CI_UPLOAD_ARTIFACT_FILES.map(
      (artifactFile) => `${REQUIRED_COMPAT_CI_UPLOAD_ARTIFACT_PATH_PREFIX}${artifactFile}`
    );
    expect(artifactName).toBe(REQUIRED_COMPAT_CI_UPLOAD_ARTIFACT_NAME);
    expect(ifNoFilesFoundValue).toBe(REQUIRED_COMPAT_CI_UPLOAD_IF_NO_FILES_FOUND);
    expect(uploadPaths).toEqual(expectedPaths);
  });

  it('keeps failure artifact upload step aligned with compat report directory contracts', () => {
    const workflowYaml = loadCiWorkflowYaml();
    const stepBlock = extractStepBlock(workflowYaml, REQUIRED_COMPAT_CI_FAILURE_UPLOAD_STEP_NAME);
    const ifCondition = extractScalarField(stepBlock, 'if', REQUIRED_COMPAT_CI_FAILURE_UPLOAD_STEP_NAME);
    const artifactName = extractScalarField(stepBlock, 'name', REQUIRED_COMPAT_CI_FAILURE_UPLOAD_STEP_NAME);
    const uploadPath = extractScalarField(stepBlock, 'path', REQUIRED_COMPAT_CI_FAILURE_UPLOAD_STEP_NAME);
    const ifNoFilesFoundValue = extractScalarField(stepBlock, 'if-no-files-found', REQUIRED_COMPAT_CI_FAILURE_UPLOAD_STEP_NAME);
    expect(ifCondition).toBe(REQUIRED_COMPAT_CI_FAILURE_UPLOAD_CONDITION);
    expect(artifactName).toBe(REQUIRED_COMPAT_CI_FAILURE_UPLOAD_ARTIFACT_NAME);
    expect(uploadPath).toBe(REQUIRED_COMPAT_CI_FAILURE_UPLOAD_PATH);
    expect(ifNoFilesFoundValue).toBe(REQUIRED_COMPAT_CI_FAILURE_UPLOAD_IF_NO_FILES_FOUND);
  });
});
