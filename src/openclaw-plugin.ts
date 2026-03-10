const clawvaultPlugin = {
  id: "clawvault",
  name: "ClawVault",
  description: "Structured memory system for AI agents with context death resilience",
  register() {
    // Hooks and CLI remain package-driven; this entry exists so OpenClaw can
    // load the plugin manifest/config schema and track the package canonically.
  },
};

export default clawvaultPlugin;
