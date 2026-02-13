export const REQUIRED_COMPAT_ARTIFACT_BUNDLE_ARTIFACT_NAMES = Object.freeze([
  'summary.json',
  'report-schema-validator-result.json',
  'validator-result.json',
  'schema-validator-result.json',
  'validator-result-verifier-result.json',
  'artifact-bundle-manifest-validator-result.json'
]);

export const REQUIRED_COMPAT_ARTIFACT_BUNDLE_ARTIFACT_COUNT = REQUIRED_COMPAT_ARTIFACT_BUNDLE_ARTIFACT_NAMES.length;

export const REQUIRED_COMPAT_ARTIFACT_BUNDLE_ARTIFACT_FILES = Object.freeze(
  Object.fromEntries(
    REQUIRED_COMPAT_ARTIFACT_BUNDLE_ARTIFACT_NAMES.map((artifactName) => [artifactName, artifactName])
  )
);

export const REQUIRED_COMPAT_ARTIFACT_BUNDLE_VERSION_FIELDS = Object.freeze({
  'summary.json': 'summarySchemaVersion',
  'report-schema-validator-result.json': 'outputSchemaVersion',
  'validator-result.json': 'outputSchemaVersion',
  'schema-validator-result.json': 'outputSchemaVersion',
  'validator-result-verifier-result.json': 'outputSchemaVersion',
  'artifact-bundle-manifest-validator-result.json': 'outputSchemaVersion'
});

export const REQUIRED_COMPAT_ARTIFACT_BUNDLE_PATH_FIELDS = Object.freeze([
  Object.freeze({ fieldName: 'summaryPath', artifactName: 'summary.json' }),
  Object.freeze({ fieldName: 'validatorResultPath', artifactName: 'validator-result.json' }),
  Object.freeze({ fieldName: 'reportSchemaValidatorResultPath', artifactName: 'report-schema-validator-result.json' }),
  Object.freeze({ fieldName: 'schemaValidatorResultPath', artifactName: 'schema-validator-result.json' }),
  Object.freeze({ fieldName: 'validatorResultVerifierResultPath', artifactName: 'validator-result-verifier-result.json' }),
  Object.freeze({ fieldName: 'artifactBundleManifestValidatorResultPath', artifactName: 'artifact-bundle-manifest-validator-result.json' })
]);
