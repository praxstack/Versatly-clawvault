/**
 * Quick checkpoint command - fast state save for context death resilience
 */
interface CheckpointOptions {
    workingOn?: string;
    focus?: string;
    blocked?: string;
    vaultPath: string;
}
interface CheckpointData {
    timestamp: string;
    workingOn: string | null;
    focus: string | null;
    blocked: string | null;
    sessionId?: string;
}
declare function flush(): Promise<CheckpointData | null>;
declare function checkpoint(options: CheckpointOptions): Promise<CheckpointData>;
declare function clearDirtyFlag(vaultPath: string): Promise<void>;
declare function checkDirtyDeath(vaultPath: string): Promise<{
    died: boolean;
    checkpoint: CheckpointData | null;
    deathTime: string | null;
}>;
declare function setSessionState(vaultPath: string, sessionId: string): Promise<void>;

export { type CheckpointData, type CheckpointOptions, checkDirtyDeath, checkpoint, clearDirtyFlag, flush, setSessionState };
