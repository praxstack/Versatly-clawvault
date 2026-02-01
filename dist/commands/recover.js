import {
  checkDirtyDeath,
  clearDirtyFlag
} from "../chunk-NITF7AHR.js";

// src/commands/recover.ts
import * as fs from "fs";
import * as path from "path";
async function recover(vaultPath, clearFlag = false) {
  const { died, checkpoint, deathTime } = await checkDirtyDeath(vaultPath);
  if (!died) {
    return {
      died: false,
      deathTime: null,
      checkpoint: null,
      handoffPath: null,
      handoffContent: null,
      recoveryMessage: "No context death detected. Clean startup."
    };
  }
  const handoffsDir = path.join(vaultPath, "handoffs");
  let handoffPath = null;
  let handoffContent = null;
  if (fs.existsSync(handoffsDir)) {
    const files = fs.readdirSync(handoffsDir).filter((f) => f.startsWith("handoff-") && f.endsWith(".md")).sort().reverse();
    if (files.length > 0) {
      handoffPath = path.join(handoffsDir, files[0]);
      handoffContent = fs.readFileSync(handoffPath, "utf-8");
    }
  }
  let message = "\u26A0\uFE0F **CONTEXT DEATH DETECTED**\n\n";
  message += `Your previous session died at ${deathTime}.

`;
  if (checkpoint) {
    message += "**Last known state:**\n";
    if (checkpoint.workingOn) {
      message += `- Working on: ${checkpoint.workingOn}
`;
    }
    if (checkpoint.focus) {
      message += `- Focus: ${checkpoint.focus}
`;
    }
    if (checkpoint.blocked) {
      message += `- Blocked: ${checkpoint.blocked}
`;
    }
    message += "\n";
  }
  if (handoffPath) {
    message += `**Last handoff:** ${path.basename(handoffPath)}
`;
    message += "Review and resume from where you left off.\n";
  } else {
    message += "**No handoff found.** You may have lost context.\n";
  }
  if (clearFlag) {
    await clearDirtyFlag(vaultPath);
  }
  return {
    died: true,
    deathTime,
    checkpoint,
    handoffPath,
    handoffContent,
    recoveryMessage: message
  };
}
function formatRecoveryInfo(info) {
  if (!info.died) {
    return "\u2713 Clean startup - no context death detected.";
  }
  let output = "\n\u26A0\uFE0F  CONTEXT DEATH DETECTED\n";
  output += "\u2550".repeat(40) + "\n\n";
  output += `Death time: ${info.deathTime}

`;
  if (info.checkpoint) {
    output += "Last checkpoint:\n";
    if (info.checkpoint.workingOn) {
      output += `  \u2022 Working on: ${info.checkpoint.workingOn}
`;
    }
    if (info.checkpoint.focus) {
      output += `  \u2022 Focus: ${info.checkpoint.focus}
`;
    }
    if (info.checkpoint.blocked) {
      output += `  \u2022 Blocked: ${info.checkpoint.blocked}
`;
    }
    output += "\n";
  }
  if (info.handoffPath) {
    output += `Last handoff: ${path.basename(info.handoffPath)}
`;
  } else {
    output += "No handoff found - context may be lost.\n";
  }
  output += "\n" + "\u2550".repeat(40) + "\n";
  output += "Run `clawvault recap` to see full context.\n";
  return output;
}
export {
  formatRecoveryInfo,
  recover
};
