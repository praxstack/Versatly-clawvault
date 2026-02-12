import { Command } from 'commander';

interface ObserveCommandOptions {
    watch?: string;
    threshold?: number;
    reflectThreshold?: number;
    model?: string;
    compress?: string;
    daemon?: boolean;
    vaultPath?: string;
}
declare function observeCommand(options: ObserveCommandOptions): Promise<void>;
declare function registerObserveCommand(program: Command): void;

export { type ObserveCommandOptions, observeCommand, registerObserveCommand };
