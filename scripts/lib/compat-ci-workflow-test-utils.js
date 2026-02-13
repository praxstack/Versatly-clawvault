export function escapeRegex(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

export function extractWorkflowName(workflowYaml) {
  const match = workflowYaml.match(/^name:\s*(.+)\s*$/m);
  return match ? match[1].trim() : null;
}

export function extractTopLevelFieldNames(workflowYaml) {
  return workflowYaml
    .split('\n')
    .map((line) => line.trimEnd())
    .filter((line) => /^[A-Za-z0-9_-]+:\s*/.test(line))
    .map((line) => line.replace(/^([A-Za-z0-9_-]+):.*/, '$1'));
}

export function extractOnTriggerNames(workflowYaml) {
  const lines = workflowYaml.split('\n');
  const onLineIndex = lines.findIndex((line) => /^on:\s*$/.test(line.trim()));
  if (onLineIndex < 0) {
    return null;
  }
  const triggerNames = [];
  for (let index = onLineIndex + 1; index < lines.length; index += 1) {
    const line = lines[index];
    if (/^[^\s].*:\s*$/.test(line)) {
      break;
    }
    const triggerMatch = /^  ([A-Za-z0-9_-]+):\s*$/.exec(line);
    if (triggerMatch) {
      triggerNames.push(triggerMatch[1]);
    }
  }
  return triggerNames;
}

export function extractOnTriggerSectionFieldNames(workflowYaml, triggerName) {
  const lines = workflowYaml.split('\n');
  const onLineIndex = lines.findIndex((line) => /^on:\s*$/.test(line.trim()));
  if (onLineIndex < 0) {
    return null;
  }

  let triggerLineIndex = -1;
  for (let index = onLineIndex + 1; index < lines.length; index += 1) {
    const line = lines[index];
    if (/^[^\s].*:\s*$/.test(line)) {
      break;
    }
    const triggerMatch = /^  ([A-Za-z0-9_-]+):\s*$/.exec(line);
    if (triggerMatch && triggerMatch[1] === triggerName) {
      triggerLineIndex = index;
      break;
    }
  }
  if (triggerLineIndex < 0) {
    return null;
  }

  const triggerIndent = countLeadingSpaces(lines[triggerLineIndex]);
  const triggerFieldIndent = triggerIndent + 2;
  const triggerFieldPattern = new RegExp(`^\\s{${triggerFieldIndent}}([A-Za-z0-9_-]+):\\s*`);
  const fieldNames = [];
  for (let index = triggerLineIndex + 1; index < lines.length; index += 1) {
    const line = lines[index];
    if (!line.trim()) {
      continue;
    }
    const lineIndent = countLeadingSpaces(line);
    if (lineIndent <= triggerIndent) {
      break;
    }
    const fieldMatch = triggerFieldPattern.exec(line);
    if (fieldMatch) {
      fieldNames.push(fieldMatch[1]);
    }
  }
  return fieldNames;
}

export function extractTopLevelJobNames(workflowYaml) {
  const lines = workflowYaml.split('\n');
  const jobsLineIndex = lines.findIndex((line) => /^jobs:\s*$/.test(line.trim()));
  if (jobsLineIndex < 0) {
    return null;
  }
  const jobNames = [];
  for (let index = jobsLineIndex + 1; index < lines.length; index += 1) {
    const line = lines[index];
    if (/^[^\s].*:\s*$/.test(line)) {
      break;
    }
    const jobMatch = /^  ([A-Za-z0-9_-]+):\s*$/.exec(line);
    if (jobMatch) {
      jobNames.push(jobMatch[1]);
    }
  }
  return jobNames;
}

export function countTopLevelFieldOccurrences(workflowYaml, fieldName) {
  const pattern = new RegExp(`^${escapeRegex(fieldName)}:\\s*`, 'gm');
  return [...workflowYaml.matchAll(pattern)].length;
}

export function extractPushBranches(workflowYaml) {
  const branchesBlockMatch = workflowYaml.match(/\n\s{2}push:\s*\n\s{4}branches:\s*\n((?:\s{6}- .*\n)+)/);
  if (!branchesBlockMatch) {
    return null;
  }
  return branchesBlockMatch[1]
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line.startsWith('- '))
    .map((line) => line.slice(2).trim());
}

export function hasPullRequestTrigger(workflowYaml) {
  return /\n\s{2}pull_request:\s*(?:\n|$)/.test(workflowYaml);
}

export function extractJobMetadata(workflowYaml, jobName) {
  const jobHeaderPattern = new RegExp(`\\n\\s{2}${escapeRegex(jobName)}:\\s*\\n`);
  const headerMatch = jobHeaderPattern.exec(workflowYaml);
  if (!headerMatch) {
    return null;
  }
  const startIndex = headerMatch.index;
  const remainder = workflowYaml.slice(startIndex + headerMatch[0].length);
  const nextJobMatch = /\n  [A-Za-z0-9_-]+:\s*\n/.exec(remainder);
  return {
    startIndex,
    block: nextJobMatch
      ? workflowYaml.slice(startIndex, startIndex + headerMatch[0].length + nextJobMatch.index)
      : workflowYaml.slice(startIndex)
  };
}

export function extractJobBlock(workflowYaml, jobName) {
  return extractJobMetadata(workflowYaml, jobName)?.block ?? null;
}

export function extractJobTopLevelFieldNames(jobBlock) {
  const lines = jobBlock.split('\n');
  const jobHeaderLine = lines.find((line) => /^\s{2}[A-Za-z0-9_-]+:\s*$/.test(line));
  if (!jobHeaderLine) {
    return null;
  }
  const jobHeaderIndent = countLeadingSpaces(jobHeaderLine);
  const jobFieldIndent = jobHeaderIndent + 2;
  const jobFieldPattern = new RegExp(`^\\s{${jobFieldIndent}}([A-Za-z0-9-]+):\\s*`);
  const fieldNames = [];
  for (const line of lines) {
    const fieldMatch = jobFieldPattern.exec(line);
    if (fieldMatch) {
      fieldNames.push(fieldMatch[1]);
    }
  }
  return fieldNames;
}

export function countJobNameOccurrences(workflowYaml, jobName) {
  const jobHeaderPattern = new RegExp(`\\n\\s{2}${escapeRegex(jobName)}:\\s*\\n`, 'g');
  return [...workflowYaml.matchAll(jobHeaderPattern)].length;
}

export function countStepNameOccurrences(workflowYamlOrJobBlock, stepName) {
  const stepHeaderPattern = new RegExp(`\\n\\s+- name:\\s+${escapeRegex(stepName)}\\s*\\n`, 'g');
  return [...workflowYamlOrJobBlock.matchAll(stepHeaderPattern)].length;
}

export function extractStepNames(workflowYamlOrJobBlock) {
  const stepHeaderPattern = /\n\s+- name:\s+(.+)\s*\n/g;
  return [...workflowYamlOrJobBlock.matchAll(stepHeaderPattern)].map((match) => match[1].trim());
}

export function extractStepFieldNames(stepBlock) {
  const lines = stepBlock.split('\n');
  const stepHeaderLine = lines.find((line) => line.trim().startsWith('- name:'));
  if (!stepHeaderLine) {
    return null;
  }
  const stepHeaderIndentMatch = /^(\s*)- name:\s+/.exec(stepHeaderLine);
  if (!stepHeaderIndentMatch) {
    return null;
  }
  const topLevelFieldIndent = stepHeaderIndentMatch[1].length + 2;
  const topLevelFieldPattern = new RegExp(`^\\s{${topLevelFieldIndent}}([A-Za-z0-9-]+):\\s*`);
  const fieldNames = ['name'];
  for (const line of lines) {
    const fieldMatch = topLevelFieldPattern.exec(line);
    if (!fieldMatch) {
      continue;
    }
    fieldNames.push(fieldMatch[1]);
  }
  return fieldNames;
}

export function countStepFieldOccurrences(stepBlock, fieldName) {
  return (extractStepFieldNames(stepBlock) ?? [])
    .filter((candidateFieldName) => candidateFieldName === fieldName)
    .length;
}

function countLeadingSpaces(line) {
  return line.length - line.trimStart().length;
}

export function extractNestedSectionFieldNames(stepBlock, sectionName) {
  const lines = stepBlock.split('\n');
  const sectionLineIndex = lines.findIndex((line) => {
    const trimmedLine = line.trim();
    return trimmedLine === `${sectionName}:` || trimmedLine === `${sectionName}: |`;
  });
  if (sectionLineIndex < 0) {
    return null;
  }
  const sectionIndent = countLeadingSpaces(lines[sectionLineIndex]);
  const sectionFieldNames = [];
  for (let index = sectionLineIndex + 1; index < lines.length; index += 1) {
    const line = lines[index];
    if (!line.trim()) {
      continue;
    }
    const lineIndent = countLeadingSpaces(line);
    if (lineIndent <= sectionIndent) {
      break;
    }
    const fieldMatch = /^\s*([A-Za-z0-9_-]+):\s*/.exec(line.trim());
    if (fieldMatch) {
      sectionFieldNames.push(fieldMatch[1]);
    }
  }
  return sectionFieldNames;
}

export function extractNestedSectionScalarFieldValue(stepBlock, sectionName, fieldName) {
  const lines = stepBlock.split('\n');
  const sectionLineIndex = lines.findIndex((line) => {
    const trimmedLine = line.trim();
    return trimmedLine === `${sectionName}:` || trimmedLine === `${sectionName}: |`;
  });
  if (sectionLineIndex < 0) {
    return null;
  }
  const sectionIndent = countLeadingSpaces(lines[sectionLineIndex]);
  for (let index = sectionLineIndex + 1; index < lines.length; index += 1) {
    const line = lines[index];
    if (!line.trim()) {
      continue;
    }
    const lineIndent = countLeadingSpaces(line);
    if (lineIndent <= sectionIndent) {
      break;
    }
    const fieldMatch = /^\s*([A-Za-z0-9_-]+):\s*(.*)$/.exec(line.trim());
    if (fieldMatch && fieldMatch[1] === fieldName) {
      return fieldMatch[2].trim();
    }
  }
  return null;
}

export function countScalarFieldOccurrences(block, fieldName) {
  const fieldPattern = new RegExp(`\\n\\s*${escapeRegex(fieldName)}:\\s*`, 'g');
  return [...block.matchAll(fieldPattern)].length;
}

export function extractStepMetadata(workflowYaml, stepName) {
  const stepHeaderPattern = new RegExp(`\\n\\s+- name:\\s+${escapeRegex(stepName)}\\s*\\n`);
  const headerMatch = stepHeaderPattern.exec(workflowYaml);
  if (!headerMatch) {
    return null;
  }
  const startIndex = headerMatch.index;
  const nextStepIndex = workflowYaml.indexOf('\n      - name:', startIndex + headerMatch[0].length);
  return {
    startIndex,
    block: nextStepIndex < 0 ? workflowYaml.slice(startIndex) : workflowYaml.slice(startIndex, nextStepIndex)
  };
}

export function extractStepBlock(workflowYaml, stepName) {
  return extractStepMetadata(workflowYaml, stepName)?.block ?? null;
}

export function extractScalarField(stepBlock, fieldName) {
  const line = stepBlock
    .split('\n')
    .map((entry) => entry.trim())
    .find((entry) => entry.startsWith(`${fieldName}:`));
  if (!line) {
    return null;
  }
  return line.replace(new RegExp(`^${escapeRegex(fieldName)}:\\s*`), '').trim();
}

export function extractRunCommand(stepBlock) {
  return extractScalarField(stepBlock, 'run');
}

export function extractEnvField(stepBlock, envKey) {
  return extractScalarField(stepBlock, envKey);
}

export function extractUsesField(stepBlock) {
  return extractScalarField(stepBlock, 'uses');
}

export function extractUploadArtifactPaths(stepBlock) {
  const lines = stepBlock.split('\n').map((line) => line.trim());
  const pathLineIndex = lines.findIndex((line) => line === 'path:' || line === 'path: |');
  if (pathLineIndex < 0) {
    return null;
  }
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

export function buildWorkflowContractSnapshot({
  workflowYaml,
  jobName,
  stepNames
}) {
  const normalizedStepNames = Array.isArray(stepNames) && stepNames.length > 0
    ? stepNames
    : [];
  const triggerNames = extractOnTriggerNames(workflowYaml) ?? [];
  const triggerSectionFieldNamesByTrigger = Object.fromEntries(
    triggerNames.map((triggerName) => [
      triggerName,
      extractOnTriggerSectionFieldNames(workflowYaml, triggerName)
    ])
  );
  const jobBlock = extractJobBlock(workflowYaml, jobName);
  const stepTopLevelFieldNamesByName = {};
  const stepWithFieldNamesByName = {};
  const stepEnvFieldNamesByName = {};
  for (const stepName of normalizedStepNames) {
    const stepBlock = jobBlock ? extractStepBlock(jobBlock, stepName) : null;
    stepTopLevelFieldNamesByName[stepName] = stepBlock ? extractStepFieldNames(stepBlock) : null;
    stepWithFieldNamesByName[stepName] = stepBlock ? extractNestedSectionFieldNames(stepBlock, 'with') : null;
    stepEnvFieldNamesByName[stepName] = stepBlock ? extractNestedSectionFieldNames(stepBlock, 'env') : null;
  }

  return {
    workflowName: extractWorkflowName(workflowYaml),
    topLevelFieldNames: extractTopLevelFieldNames(workflowYaml),
    triggerNames,
    triggerSectionFieldNamesByTrigger,
    pushBranches: extractPushBranches(workflowYaml),
    pullRequestTrigger: hasPullRequestTrigger(workflowYaml),
    jobNames: extractTopLevelJobNames(workflowYaml),
    jobName,
    jobTopLevelFieldNames: jobBlock ? extractJobTopLevelFieldNames(jobBlock) : null,
    jobRunsOn: jobBlock ? extractScalarField(jobBlock, 'runs-on') : null,
    jobTimeoutMinutes: jobBlock ? extractScalarField(jobBlock, 'timeout-minutes') : null,
    stepNames: jobBlock ? extractStepNames(jobBlock) : null,
    stepTopLevelFieldNamesByName,
    stepWithFieldNamesByName,
    stepEnvFieldNamesByName
  };
}
