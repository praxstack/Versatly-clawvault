import * as fs from 'fs';

export const COMPAT_ARTIFACT_BUNDLE_VALIDATOR_OUTPUT_SCHEMA_VERSION = 1;

const VALID_SUMMARY_MODES = new Set(['contract', 'fixtures']);
const VALID_VERSION_FIELDS = new Set(['summarySchemaVersion', 'outputSchemaVersion']);

function assertNonEmptyString(value, fieldName) {
  if (typeof value !== 'string' || value.length === 0) {
    throw new Error(`compat artifact bundle validator payload field "${fieldName}" must be a non-empty string`);
  }
}

function assertUniqueNonEmptyStringArray(values, fieldName) {
  if (!Array.isArray(values) || values.length === 0) {
    throw new Error(`compat artifact bundle validator payload field "${fieldName}" must be a non-empty array`);
  }
  if (values.some((value) => typeof value !== 'string' || value.length === 0)) {
    throw new Error(`compat artifact bundle validator payload field "${fieldName}" must contain only non-empty strings`);
  }
  const duplicates = values
    .filter((value, index, allValues) => allValues.indexOf(value) !== index)
    .filter((value, index, allValues) => allValues.indexOf(value) === index);
  if (duplicates.length > 0) {
    throw new Error(`compat artifact bundle validator payload field "${fieldName}" contains duplicates: ${duplicates.join(', ')}`);
  }
}

function assertNonNegativeInteger(value, fieldName) {
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`compat artifact bundle validator payload field "${fieldName}" must be a non-negative integer`);
  }
}

function ensureArtifactContractShape(entry, index) {
  if (!entry || typeof entry !== 'object' || Array.isArray(entry)) {
    throw new Error(`compat artifact bundle validator payload artifactContracts[${index}] must be an object`);
  }
  assertNonEmptyString(entry.artifactName, `artifactContracts[${index}].artifactName`);
  assertNonEmptyString(entry.artifactPath, `artifactContracts[${index}].artifactPath`);
  assertNonEmptyString(entry.schemaPath, `artifactContracts[${index}].schemaPath`);
  assertNonEmptyString(entry.schemaId, `artifactContracts[${index}].schemaId`);
  if (!VALID_VERSION_FIELDS.has(entry.versionField)) {
    throw new Error(`compat artifact bundle validator payload artifactContracts[${index}].versionField has invalid value`);
  }
  assertNonNegativeInteger(entry.expectedSchemaVersion, `artifactContracts[${index}].expectedSchemaVersion`);
  assertNonNegativeInteger(entry.actualSchemaVersion, `artifactContracts[${index}].actualSchemaVersion`);
}

export function ensureCompatArtifactBundleValidatorPayloadShape(payload) {
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    throw new Error('compat artifact bundle validator payload must be an object');
  }
  if (payload.outputSchemaVersion !== COMPAT_ARTIFACT_BUNDLE_VALIDATOR_OUTPUT_SCHEMA_VERSION) {
    throw new Error(
      `compat artifact bundle validator payload outputSchemaVersion must be ${COMPAT_ARTIFACT_BUNDLE_VALIDATOR_OUTPUT_SCHEMA_VERSION}`
    );
  }
  if (payload.status !== 'ok' && payload.status !== 'error') {
    throw new Error('compat artifact bundle validator payload status must be "ok" or "error"');
  }

  if (payload.status === 'ok') {
    assertNonEmptyString(payload.reportDir, 'reportDir');
    if (!VALID_SUMMARY_MODES.has(payload.summaryMode)) {
      throw new Error('compat artifact bundle validator payload field "summaryMode" must be "contract" or "fixtures"');
    }
    if (typeof payload.requireOk !== 'boolean') {
      throw new Error('compat artifact bundle validator payload field "requireOk" must be a boolean');
    }

    const pathFields = [
      'summaryPath',
      'validatorResultPath',
      'reportSchemaValidatorResultPath',
      'schemaValidatorResultPath',
      'validatorResultVerifierResultPath',
      'artifactBundleManifestValidatorResultPath'
    ];
    for (const fieldName of pathFields) {
      assertNonEmptyString(payload[fieldName], fieldName);
    }

    assertUniqueNonEmptyStringArray(payload.verifiedArtifacts, 'verifiedArtifacts');
    if (!Array.isArray(payload.artifactContracts) || payload.artifactContracts.length === 0) {
      throw new Error('compat artifact bundle validator payload field "artifactContracts" must be a non-empty array');
    }
    for (const [index, entry] of payload.artifactContracts.entries()) {
      ensureArtifactContractShape(entry, index);
    }
    const artifactNames = payload.artifactContracts.map((entry) => entry.artifactName);
    const duplicateArtifactNames = artifactNames
      .filter((value, index, allValues) => allValues.indexOf(value) !== index)
      .filter((value, index, allValues) => allValues.indexOf(value) === index);
    if (duplicateArtifactNames.length > 0) {
      throw new Error(`compat artifact bundle validator payload artifactContracts contains duplicates: ${duplicateArtifactNames.join(', ')}`);
    }
    return;
  }

  assertNonEmptyString(payload.error, 'error');
}

export function buildCompatArtifactBundleValidatorSuccessPayload({
  reportDir,
  summaryMode,
  requireOk,
  summaryPath,
  validatorResultPath,
  reportSchemaValidatorResultPath,
  schemaValidatorResultPath,
  validatorResultVerifierResultPath,
  artifactBundleManifestValidatorResultPath,
  verifiedArtifacts,
  artifactContracts
}) {
  const payload = {
    outputSchemaVersion: COMPAT_ARTIFACT_BUNDLE_VALIDATOR_OUTPUT_SCHEMA_VERSION,
    status: 'ok',
    reportDir,
    summaryMode,
    requireOk,
    summaryPath,
    validatorResultPath,
    reportSchemaValidatorResultPath,
    schemaValidatorResultPath,
    validatorResultVerifierResultPath,
    artifactBundleManifestValidatorResultPath,
    verifiedArtifacts,
    artifactContracts
  };
  ensureCompatArtifactBundleValidatorPayloadShape(payload);
  return payload;
}

export function buildCompatArtifactBundleValidatorErrorPayload(error) {
  const payload = {
    outputSchemaVersion: COMPAT_ARTIFACT_BUNDLE_VALIDATOR_OUTPUT_SCHEMA_VERSION,
    status: 'error',
    error: String(error ?? '')
  };
  ensureCompatArtifactBundleValidatorPayloadShape(payload);
  return payload;
}

export function loadCompatArtifactBundleValidatorPayload(payloadPath) {
  try {
    const raw = fs.readFileSync(payloadPath, 'utf-8');
    const parsed = JSON.parse(raw);
    ensureCompatArtifactBundleValidatorPayloadShape(parsed);
    return parsed;
  } catch (err) {
    throw new Error(`Unable to read compat artifact bundle validator payload at ${payloadPath}: ${err?.message || String(err)}`);
  }
}
