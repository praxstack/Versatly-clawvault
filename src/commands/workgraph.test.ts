/**
 * Tests for workgraph CLI commands.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import * as registry from '../workgraph/registry.js';
import * as ledger from '../workgraph/ledger.js';
import * as store from '../workgraph/store.js';
import * as thread from '../workgraph/thread.js';

function makeTempDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'clawvault-wg-'));
}

describe('workgraph commands', () => {
  let tempDir: string;

  beforeEach(() => {
    tempDir = makeTempDir();
  });

  afterEach(() => {
    fs.rmSync(tempDir, { recursive: true, force: true });
  });

  describe('thread lifecycle', () => {
    it('creates a thread with default fields', () => {
      const inst = thread.createThread(tempDir, 'Test Thread', 'Complete the test', 'test-agent');

      expect(inst.path).toBe('threads/test-thread.md');
      expect(inst.type).toBe('thread');
      expect(inst.fields.title).toBe('Test Thread');
      expect(inst.fields.goal).toBe('Complete the test');
      expect(inst.fields.status).toBe('open');
      expect(inst.fields.priority).toBe('medium');
      expect(inst.fields.deps).toEqual([]);
      expect(inst.fields.tags).toEqual([]);
    });

    it('creates a thread with custom options', () => {
      const inst = thread.createThread(tempDir, 'Urgent Task', 'Fix the bug', 'test-agent', {
        priority: 'urgent',
        deps: ['threads/other.md'],
        tags: ['bug', 'critical'],
      });

      expect(inst.fields.priority).toBe('urgent');
      expect(inst.fields.deps).toEqual(['threads/other.md']);
      expect(inst.fields.tags).toEqual(['bug', 'critical']);
    });

    it('claims an open thread', () => {
      thread.createThread(tempDir, 'Claimable', 'Do work', 'creator');
      const claimed = thread.claim(tempDir, 'threads/claimable.md', 'claimer');

      expect(claimed.fields.status).toBe('active');
      expect(claimed.fields.owner).toBe('claimer');
    });

    it('prevents claiming an already claimed thread', () => {
      thread.createThread(tempDir, 'Exclusive', 'Work', 'creator');
      thread.claim(tempDir, 'threads/exclusive.md', 'agent-1');

      expect(() => {
        thread.claim(tempDir, 'threads/exclusive.md', 'agent-2');
      }).toThrow(/Cannot claim thread/);
    });

    it('releases a claimed thread', () => {
      thread.createThread(tempDir, 'Releasable', 'Work', 'creator');
      thread.claim(tempDir, 'threads/releasable.md', 'agent');
      const released = thread.release(tempDir, 'threads/releasable.md', 'agent');

      expect(released.fields.status).toBe('open');
      expect(released.fields.owner).toBeNull();
    });

    it('blocks a thread with dependency', () => {
      thread.createThread(tempDir, 'Blockable', 'Work', 'creator');
      thread.claim(tempDir, 'threads/blockable.md', 'agent');
      const blocked = thread.block(tempDir, 'threads/blockable.md', 'agent', 'threads/dependency.md');

      expect(blocked.fields.status).toBe('blocked');
      expect(blocked.fields.deps).toContain('threads/dependency.md');
    });

    it('completes a thread with output', () => {
      thread.createThread(tempDir, 'Completable', 'Work', 'creator');
      thread.claim(tempDir, 'threads/completable.md', 'agent');
      const done = thread.done(tempDir, 'threads/completable.md', 'agent', 'Task completed successfully');

      expect(done.fields.status).toBe('done');
      expect(done.body).toContain('Task completed successfully');
    });

    it('decomposes a thread into sub-threads', () => {
      thread.createThread(tempDir, 'Parent Task', 'Big work', 'creator');
      const children = thread.decompose(
        tempDir,
        'threads/parent-task.md',
        [
          { title: 'Sub Task 1', goal: 'First part' },
          { title: 'Sub Task 2', goal: 'Second part' },
        ],
        'creator'
      );

      expect(children).toHaveLength(2);
      expect(children[0].fields.title).toBe('Sub Task 1');
      expect(children[0].fields.parent).toBe('threads/parent-task.md');
      expect(children[1].fields.title).toBe('Sub Task 2');

      const parent = store.read(tempDir, 'threads/parent-task.md');
      expect(parent?.body).toContain('Sub-threads');
    });
  });

  describe('ledger operations', () => {
    it('records thread creation', () => {
      thread.createThread(tempDir, 'Logged', 'Work', 'agent');
      const entries = ledger.readAll(tempDir);

      expect(entries.length).toBeGreaterThan(0);
      const createEntry = entries.find(e => e.op === 'create');
      expect(createEntry).toBeDefined();
      expect(createEntry?.target).toBe('threads/logged.md');
      expect(createEntry?.actor).toBe('agent');
    });

    it('records claim and release', () => {
      thread.createThread(tempDir, 'Tracked', 'Work', 'creator');
      thread.claim(tempDir, 'threads/tracked.md', 'claimer');
      thread.release(tempDir, 'threads/tracked.md', 'claimer');

      const entries = ledger.readAll(tempDir);
      const claimEntry = entries.find(e => e.op === 'claim');
      const releaseEntry = entries.find(e => e.op === 'release');

      expect(claimEntry).toBeDefined();
      expect(claimEntry?.actor).toBe('claimer');
      expect(releaseEntry).toBeDefined();
    });

    it('tracks current owner correctly', () => {
      thread.createThread(tempDir, 'Ownership', 'Work', 'creator');

      expect(ledger.currentOwner(tempDir, 'threads/ownership.md')).toBeNull();

      thread.claim(tempDir, 'threads/ownership.md', 'agent-1');
      expect(ledger.currentOwner(tempDir, 'threads/ownership.md')).toBe('agent-1');

      thread.release(tempDir, 'threads/ownership.md', 'agent-1');
      expect(ledger.currentOwner(tempDir, 'threads/ownership.md')).toBeNull();
    });

    it('returns recent entries', () => {
      thread.createThread(tempDir, 'Recent 1', 'Work', 'agent');
      thread.createThread(tempDir, 'Recent 2', 'Work', 'agent');
      thread.createThread(tempDir, 'Recent 3', 'Work', 'agent');

      const recent = ledger.recent(tempDir, 2);
      expect(recent).toHaveLength(2);
    });

    it('filters by actor', () => {
      thread.createThread(tempDir, 'By Agent 1', 'Work', 'agent-1');
      thread.createThread(tempDir, 'By Agent 2', 'Work', 'agent-2');

      const agent1Activity = ledger.activityOf(tempDir, 'agent-1');
      expect(agent1Activity.every(e => e.actor === 'agent-1')).toBe(true);
    });
  });

  describe('registry operations', () => {
    it('loads built-in types', () => {
      const types = registry.listTypes(tempDir);
      const typeNames = types.map(t => t.name);

      expect(typeNames).toContain('thread');
      expect(typeNames).toContain('space');
      expect(typeNames).toContain('decision');
      expect(typeNames).toContain('lesson');
      expect(typeNames).toContain('fact');
      expect(typeNames).toContain('agent');
    });

    it('defines a custom type', () => {
      const typeDef = registry.defineType(
        tempDir,
        'custom-type',
        'A custom primitive',
        { customField: { type: 'string' } },
        'test-agent'
      );

      expect(typeDef.name).toBe('custom-type');
      expect(typeDef.builtIn).toBe(false);
      expect(typeDef.fields.customField).toBeDefined();
      expect(typeDef.directory).toBe('custom-types');
    });

    it('prevents redefining built-in types', () => {
      expect(() => {
        registry.defineType(tempDir, 'thread', 'Override', {}, 'agent');
      }).toThrow(/Cannot redefine built-in type/);
    });

    it('extends existing type with new fields', () => {
      const extended = registry.extendType(
        tempDir,
        'thread',
        { customExtension: { type: 'string', description: 'Extended field' } },
        'agent'
      );

      expect(extended.fields.customExtension).toBeDefined();
    });
  });

  describe('store operations', () => {
    it('creates primitive instances', () => {
      const inst = store.create(
        tempDir,
        'decision',
        { title: 'Important Decision', date: '2026-02-27' },
        'We decided to do X because Y.',
        'agent'
      );

      expect(inst.path).toBe('decisions/important-decision.md');
      expect(inst.type).toBe('decision');
      expect(inst.fields.title).toBe('Important Decision');
    });

    it('reads primitive instances', () => {
      store.create(tempDir, 'lesson', { title: 'Learned Something' }, 'Content', 'agent');
      const read = store.read(tempDir, 'lessons/learned-something.md');

      expect(read).not.toBeNull();
      expect(read?.fields.title).toBe('Learned Something');
    });

    it('lists primitives by type', () => {
      store.create(tempDir, 'fact', { title: 'Fact One', subject: 'A', predicate: 'is', object: 'B' }, '', 'agent');
      store.create(tempDir, 'fact', { title: 'Fact Two', subject: 'C', predicate: 'has', object: 'D' }, '', 'agent');

      const facts = store.list(tempDir, 'fact');
      expect(facts).toHaveLength(2);
    });

    it('updates primitive fields', () => {
      store.create(tempDir, 'decision', { title: 'Updatable', status: 'active' }, '', 'agent');
      const updated = store.update(
        tempDir,
        'decisions/updatable.md',
        { status: 'superseded' },
        undefined,
        'agent'
      );

      expect(updated.fields.status).toBe('superseded');
    });

    it('finds primitives by field value', () => {
      thread.createThread(tempDir, 'Open 1', 'Work', 'agent');
      thread.createThread(tempDir, 'Open 2', 'Work', 'agent');
      thread.createThread(tempDir, 'Active', 'Work', 'agent');
      thread.claim(tempDir, 'threads/active.md', 'agent');

      const openThreads = store.findByField(tempDir, 'thread', 'status', 'open');
      expect(openThreads).toHaveLength(2);

      const activeThreads = store.findByField(tempDir, 'thread', 'status', 'active');
      expect(activeThreads).toHaveLength(1);
    });
  });

  describe('thread status transitions', () => {
    it('allows valid transitions', () => {
      thread.createThread(tempDir, 'Transition Test', 'Work', 'agent');

      // open -> active (via claim)
      thread.claim(tempDir, 'threads/transition-test.md', 'agent');
      let t = store.read(tempDir, 'threads/transition-test.md');
      expect(t?.fields.status).toBe('active');

      // active -> blocked
      thread.block(tempDir, 'threads/transition-test.md', 'agent', 'blocker');
      t = store.read(tempDir, 'threads/transition-test.md');
      expect(t?.fields.status).toBe('blocked');

      // blocked -> active (via unblock)
      thread.unblock(tempDir, 'threads/transition-test.md', 'agent');
      t = store.read(tempDir, 'threads/transition-test.md');
      expect(t?.fields.status).toBe('active');

      // active -> done
      thread.done(tempDir, 'threads/transition-test.md', 'agent');
      t = store.read(tempDir, 'threads/transition-test.md');
      expect(t?.fields.status).toBe('done');
    });

    it('prevents invalid transitions', () => {
      thread.createThread(tempDir, 'Invalid Transition', 'Work', 'agent');

      // Cannot go directly from open to done
      expect(() => {
        thread.done(tempDir, 'threads/invalid-transition.md', 'agent');
      }).toThrow(/Invalid transition/);
    });
  });

  describe('claim exclusivity', () => {
    it('tracks all claims correctly', () => {
      thread.createThread(tempDir, 'Claim 1', 'Work', 'agent');
      thread.createThread(tempDir, 'Claim 2', 'Work', 'agent');
      thread.claim(tempDir, 'threads/claim-1.md', 'agent-1');
      thread.claim(tempDir, 'threads/claim-2.md', 'agent-2');

      const claims = ledger.allClaims(tempDir);
      expect(claims.size).toBe(2);
      expect(claims.get('threads/claim-1.md')).toBe('agent-1');
      expect(claims.get('threads/claim-2.md')).toBe('agent-2');
    });

    it('removes claim on release', () => {
      thread.createThread(tempDir, 'Release Test', 'Work', 'agent');
      thread.claim(tempDir, 'threads/release-test.md', 'agent');

      let claims = ledger.allClaims(tempDir);
      expect(claims.has('threads/release-test.md')).toBe(true);

      thread.release(tempDir, 'threads/release-test.md', 'agent');
      claims = ledger.allClaims(tempDir);
      expect(claims.has('threads/release-test.md')).toBe(false);
    });

    it('removes claim on done', () => {
      thread.createThread(tempDir, 'Done Test', 'Work', 'agent');
      thread.claim(tempDir, 'threads/done-test.md', 'agent');
      thread.done(tempDir, 'threads/done-test.md', 'agent');

      const claims = ledger.allClaims(tempDir);
      expect(claims.has('threads/done-test.md')).toBe(false);
    });
  });

  describe('helper queries', () => {
    it('finds open threads', () => {
      thread.createThread(tempDir, 'Open A', 'Work', 'agent');
      thread.createThread(tempDir, 'Open B', 'Work', 'agent');
      thread.createThread(tempDir, 'Active', 'Work', 'agent');
      thread.claim(tempDir, 'threads/active.md', 'agent');

      const open = store.openThreads(tempDir);
      expect(open).toHaveLength(2);
    });

    it('finds active threads', () => {
      thread.createThread(tempDir, 'Will Be Active', 'Work', 'agent');
      thread.claim(tempDir, 'threads/will-be-active.md', 'agent');

      const active = store.activeThreads(tempDir);
      expect(active).toHaveLength(1);
      expect(active[0].fields.title).toBe('Will Be Active');
    });

    it('finds blocked threads', () => {
      thread.createThread(tempDir, 'Will Be Blocked', 'Work', 'agent');
      thread.claim(tempDir, 'threads/will-be-blocked.md', 'agent');
      thread.block(tempDir, 'threads/will-be-blocked.md', 'agent', 'blocker');

      const blocked = store.blockedThreads(tempDir);
      expect(blocked).toHaveLength(1);
    });

    it('gets history of a target', () => {
      thread.createThread(tempDir, 'History Test', 'Work', 'agent');
      thread.claim(tempDir, 'threads/history-test.md', 'agent');
      thread.release(tempDir, 'threads/history-test.md', 'agent');

      const history = ledger.historyOf(tempDir, 'threads/history-test.md');
      expect(history.length).toBeGreaterThanOrEqual(3);

      const ops = history.map(e => e.op);
      expect(ops).toContain('create');
      expect(ops).toContain('claim');
      expect(ops).toContain('release');
    });
  });

  describe('edge cases', () => {
    it('handles empty vault gracefully', () => {
      const threads = store.list(tempDir, 'thread');
      expect(threads).toEqual([]);

      const entries = ledger.readAll(tempDir);
      expect(entries).toEqual([]);
    });

    it('returns null for non-existent primitive', () => {
      const result = store.read(tempDir, 'threads/does-not-exist.md');
      expect(result).toBeNull();
    });

    it('throws for operations on non-existent thread', () => {
      expect(() => {
        thread.claim(tempDir, 'threads/ghost.md', 'agent');
      }).toThrow(/not found/);
    });

    it('slugifies titles correctly', () => {
      const inst = thread.createThread(
        tempDir,
        'This Is A Very Long Title With Special Characters!@#$%',
        'Goal',
        'agent'
      );

      expect(inst.path).toMatch(/^threads\/this-is-a-very-long-title/);
      expect(inst.path).not.toContain('!');
      expect(inst.path).not.toContain('@');
    });
  });
});
