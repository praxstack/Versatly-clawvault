import * as fs from 'fs';
import * as path from 'path';
import chokidar, { type FSWatcher } from 'chokidar';
import type { Observer } from './observer.js';

export interface SessionWatcherOptions {
  ignoreInitial?: boolean;
  debounceMs?: number;
}

export class SessionWatcher {
  private readonly watchPath: string;
  private readonly observer: Observer;
  private readonly ignoreInitial: boolean;
  private readonly debounceMs: number;
  private watcher: FSWatcher | null = null;
  private fileOffsets = new Map<string, number>();
  private pendingPaths = new Set<string>();
  private debounceTimer: NodeJS.Timeout | null = null;
  private processingQueue: Promise<void> = Promise.resolve();

  constructor(watchPath: string, observer: Observer, options: SessionWatcherOptions = {}) {
    this.watchPath = path.resolve(watchPath);
    this.observer = observer;
    this.ignoreInitial = options.ignoreInitial ?? false;
    this.debounceMs = options.debounceMs ?? 500;
  }

  async start(): Promise<void> {
    if (!fs.existsSync(this.watchPath)) {
      throw new Error(`Watch path does not exist: ${this.watchPath}`);
    }

    this.watcher = chokidar.watch(this.watchPath, {
      persistent: true,
      ignoreInitial: this.ignoreInitial,
      awaitWriteFinish: {
        stabilityThreshold: 120,
        pollInterval: 30
      }
    });

    const enqueue = (changedPath: string): void => {
      this.pendingPaths.add(path.resolve(changedPath));
      this.scheduleDrain();
    };

    this.watcher.on('add', enqueue);
    this.watcher.on('change', enqueue);
    this.watcher.on('unlink', (deletedPath: string) => {
      const resolved = path.resolve(deletedPath);
      this.fileOffsets.delete(resolved);
      this.pendingPaths.delete(resolved);
    });

    await new Promise<void>((resolve, reject) => {
      this.watcher?.once('ready', () => resolve());
      this.watcher?.once('error', (error) => reject(error));
    });
  }

  async stop(): Promise<void> {
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
      this.debounceTimer = null;
    }
    this.pendingPaths.clear();
    await this.processingQueue.catch(() => undefined);
    await this.watcher?.close();
    this.watcher = null;
  }

  private scheduleDrain(): void {
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
    }

    this.debounceTimer = setTimeout(() => {
      this.debounceTimer = null;
      const nextPaths = [...this.pendingPaths];
      this.pendingPaths.clear();

      for (const changedPath of nextPaths) {
        this.processingQueue = this.processingQueue
          .then(() => this.consumeFile(changedPath))
          .catch(() => undefined);
      }
    }, this.debounceMs);
  }

  private async consumeFile(filePath: string): Promise<void> {
    const resolved = path.resolve(filePath);
    if (!fs.existsSync(resolved)) {
      return;
    }

    const stats = fs.statSync(resolved);
    if (!stats.isFile()) {
      return;
    }

    const previousOffset = this.fileOffsets.get(resolved) ?? 0;
    const startOffset = stats.size < previousOffset ? 0 : previousOffset;
    if (stats.size <= startOffset) {
      this.fileOffsets.set(resolved, stats.size);
      return;
    }

    const bytesToRead = stats.size - startOffset;
    const buffer = Buffer.alloc(bytesToRead);
    const fd = fs.openSync(resolved, 'r');

    try {
      fs.readSync(fd, buffer, 0, bytesToRead, startOffset);
    } finally {
      fs.closeSync(fd);
    }

    this.fileOffsets.set(resolved, stats.size);
    const chunk = buffer.toString('utf-8');
    const messages = chunk
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);

    if (messages.length === 0) {
      return;
    }

    await this.observer.processMessages(messages);
  }
}
