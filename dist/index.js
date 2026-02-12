import {
  buildSessionRecap,
  formatSessionRecapMarkdown,
  sessionRecapCommand
} from "./chunk-ZKGY7WTT.js";
import "./chunk-HRLWZGMA.js";
import {
  setupCommand
} from "./chunk-PIJGYMQZ.js";
import {
  buildTemplateVariables,
  renderTemplate
} from "./chunk-7766SIJP.js";
import {
  buildContext,
  contextCommand,
  formatContextMarkdown
} from "./chunk-XPGSEJGY.js";
import {
  ClawVault,
  createVault,
  findVault
} from "./chunk-3HFB7EMU.js";
import {
  DEFAULT_CATEGORIES,
  DEFAULT_CONFIG,
  MEMORY_TYPES,
  QMD_INSTALL_COMMAND,
  QMD_INSTALL_URL,
  QmdUnavailableError,
  SearchEngine,
  TYPE_TO_CATEGORY,
  extractTags,
  extractWikiLinks,
  hasQmd,
  qmdEmbed,
  qmdUpdate
} from "./chunk-MIIXBNO3.js";
import {
  Compressor,
  Observer,
  Reflector,
  SessionWatcher,
  observeCommand,
  registerObserveCommand
} from "./chunk-NSPPVMWC.js";

// src/index.ts
import * as fs from "fs";
function readPackageVersion() {
  try {
    const pkgUrl = new URL("../package.json", import.meta.url);
    const pkg = JSON.parse(fs.readFileSync(pkgUrl, "utf-8"));
    return pkg.version ?? "0.0.0";
  } catch {
    return "0.0.0";
  }
}
var VERSION = readPackageVersion();
function registerCommanderCommands(program) {
  registerObserveCommand(program);
  return program;
}
export {
  ClawVault,
  Compressor,
  DEFAULT_CATEGORIES,
  DEFAULT_CONFIG,
  MEMORY_TYPES,
  Observer,
  QMD_INSTALL_COMMAND,
  QMD_INSTALL_URL,
  QmdUnavailableError,
  Reflector,
  SearchEngine,
  SessionWatcher,
  TYPE_TO_CATEGORY,
  VERSION,
  buildContext,
  buildSessionRecap,
  buildTemplateVariables,
  contextCommand,
  createVault,
  extractTags,
  extractWikiLinks,
  findVault,
  formatContextMarkdown,
  formatSessionRecapMarkdown,
  hasQmd,
  observeCommand,
  qmdEmbed,
  qmdUpdate,
  registerCommanderCommands,
  registerObserveCommand,
  renderTemplate,
  sessionRecapCommand,
  setupCommand
};
