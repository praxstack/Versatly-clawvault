import { describe, expect, it } from 'vitest';
import {
  REQUIRED_COMPAT_ARTIFACT_BUNDLE_ARTIFACT_COUNT,
  REQUIRED_COMPAT_ARTIFACT_BUNDLE_ARTIFACT_DEFINITIONS,
  REQUIRED_COMPAT_ARTIFACT_BUNDLE_ARTIFACT_FILES,
  REQUIRED_COMPAT_ARTIFACT_BUNDLE_ARTIFACT_NAMES,
  REQUIRED_COMPAT_ARTIFACT_BUNDLE_PATH_FIELDS,
  REQUIRED_COMPAT_ARTIFACT_BUNDLE_SCHEMA_IDS,
  REQUIRED_COMPAT_ARTIFACT_BUNDLE_SCHEMA_PATHS,
  REQUIRED_COMPAT_ARTIFACT_BUNDLE_VERSION_FIELDS
} from './compat-artifact-bundle-contracts.mjs';

describe('compat artifact bundle contracts constants', () => {
  it('keeps canonical artifact definitions unique and ordered', () => {
    expect(REQUIRED_COMPAT_ARTIFACT_BUNDLE_ARTIFACT_DEFINITIONS.length).toBe(REQUIRED_COMPAT_ARTIFACT_BUNDLE_ARTIFACT_COUNT);
    const artifactNames = REQUIRED_COMPAT_ARTIFACT_BUNDLE_ARTIFACT_DEFINITIONS.map((entry) => entry.artifactName);
    expect(new Set(artifactNames).size).toBe(REQUIRED_COMPAT_ARTIFACT_BUNDLE_ARTIFACT_COUNT);
    expect(artifactNames).toEqual(REQUIRED_COMPAT_ARTIFACT_BUNDLE_ARTIFACT_NAMES);
  });

  it('keeps required artifact names unique and non-empty', () => {
    expect(REQUIRED_COMPAT_ARTIFACT_BUNDLE_ARTIFACT_NAMES.length).toBeGreaterThan(0);
    expect(REQUIRED_COMPAT_ARTIFACT_BUNDLE_ARTIFACT_COUNT).toBe(REQUIRED_COMPAT_ARTIFACT_BUNDLE_ARTIFACT_NAMES.length);
    expect(new Set(REQUIRED_COMPAT_ARTIFACT_BUNDLE_ARTIFACT_NAMES).size).toBe(
      REQUIRED_COMPAT_ARTIFACT_BUNDLE_ARTIFACT_NAMES.length
    );
    expect(REQUIRED_COMPAT_ARTIFACT_BUNDLE_ARTIFACT_NAMES.every((value) => typeof value === 'string' && value.length > 0)).toBe(true);
  });

  it('keeps required path bindings aligned with required artifact set', () => {
    const boundArtifactNames = REQUIRED_COMPAT_ARTIFACT_BUNDLE_PATH_FIELDS.map((entry) => entry.artifactName);
    expect(new Set(boundArtifactNames).size).toBe(REQUIRED_COMPAT_ARTIFACT_BUNDLE_PATH_FIELDS.length);
    expect(new Set(REQUIRED_COMPAT_ARTIFACT_BUNDLE_PATH_FIELDS.map((entry) => entry.fieldName)).size).toBe(
      REQUIRED_COMPAT_ARTIFACT_BUNDLE_PATH_FIELDS.length
    );
    expect(boundArtifactNames).toEqual(expect.arrayContaining(REQUIRED_COMPAT_ARTIFACT_BUNDLE_ARTIFACT_NAMES));
  });

  it('keeps required version-field bindings aligned with required artifact set', () => {
    expect(Object.keys(REQUIRED_COMPAT_ARTIFACT_BUNDLE_VERSION_FIELDS).sort()).toEqual(
      [...REQUIRED_COMPAT_ARTIFACT_BUNDLE_ARTIFACT_NAMES].sort()
    );
    expect(REQUIRED_COMPAT_ARTIFACT_BUNDLE_VERSION_FIELDS['summary.json']).toBe('summarySchemaVersion');
    for (const artifactName of REQUIRED_COMPAT_ARTIFACT_BUNDLE_ARTIFACT_NAMES.filter((name) => name !== 'summary.json')) {
      expect(REQUIRED_COMPAT_ARTIFACT_BUNDLE_VERSION_FIELDS[artifactName]).toBe('outputSchemaVersion');
    }
  });

  it('keeps required artifact-file bindings aligned with required artifact set', () => {
    expect(Object.keys(REQUIRED_COMPAT_ARTIFACT_BUNDLE_ARTIFACT_FILES).sort()).toEqual(
      [...REQUIRED_COMPAT_ARTIFACT_BUNDLE_ARTIFACT_NAMES].sort()
    );
    for (const artifactName of REQUIRED_COMPAT_ARTIFACT_BUNDLE_ARTIFACT_NAMES) {
      expect(REQUIRED_COMPAT_ARTIFACT_BUNDLE_ARTIFACT_FILES[artifactName]).toBe(artifactName);
    }
  });

  it('keeps required schema path/id bindings aligned with required artifact set', () => {
    expect(Object.keys(REQUIRED_COMPAT_ARTIFACT_BUNDLE_SCHEMA_PATHS).sort()).toEqual(
      [...REQUIRED_COMPAT_ARTIFACT_BUNDLE_ARTIFACT_NAMES].sort()
    );
    expect(Object.keys(REQUIRED_COMPAT_ARTIFACT_BUNDLE_SCHEMA_IDS).sort()).toEqual(
      [...REQUIRED_COMPAT_ARTIFACT_BUNDLE_ARTIFACT_NAMES].sort()
    );
    for (const artifactName of REQUIRED_COMPAT_ARTIFACT_BUNDLE_ARTIFACT_NAMES) {
      const schemaPath = REQUIRED_COMPAT_ARTIFACT_BUNDLE_SCHEMA_PATHS[artifactName];
      const schemaId = REQUIRED_COMPAT_ARTIFACT_BUNDLE_SCHEMA_IDS[artifactName];
      expect(schemaPath.startsWith('schemas/')).toBe(true);
      expect(schemaId).toBe(`https://clawvault.dev/${schemaPath}`);
    }
  });
});
