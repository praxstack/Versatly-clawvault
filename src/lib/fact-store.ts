/**
 * Fact Store — persistent storage + conflict resolution for extracted facts.
 *
 * Facts are stored in .clawvault/facts.jsonl (append-only log).
 * An in-memory index enables fast lookup by entity, relation, category.
 * Conflict resolution: when a new fact matches an existing one by entity+relation,
 * the old fact gets validUntil set and the new fact replaces it.
 */

import * as fs from 'fs';
import * as path from 'path';
import { ExtractedFact, normalizeEntity } from './fact-extractor.js';

export interface FactStoreStats {
  totalFacts: number;
  activeFacts: number;
  supersededFacts: number;
  entities: number;
  relations: number;
}

export class FactStore {
  private facts: Map<string, ExtractedFact> = new Map();
  private byEntity: Map<string, Set<string>> = new Map();
  private byRelation: Map<string, Set<string>> = new Map();
  private byCategory: Map<string, Set<string>> = new Map();
  private factsPath: string;
  private dirty = false;

  constructor(vaultPath: string) {
    this.factsPath = path.join(vaultPath, '.clawvault', 'facts.jsonl');
  }

  /** Load facts from disk */
  load(): void {
    this.facts.clear();
    this.byEntity.clear();
    this.byRelation.clear();
    this.byCategory.clear();

    if (!fs.existsSync(this.factsPath)) return;

    const lines = fs.readFileSync(this.factsPath, 'utf-8').split('\n').filter(l => l.trim());
    for (const line of lines) {
      try {
        const fact = JSON.parse(line) as ExtractedFact;
        this.indexFact(fact);
      } catch {
        // Skip malformed lines
      }
    }
  }

  /** Add facts with conflict resolution. Returns number of conflicts resolved. */
  addFacts(newFacts: ExtractedFact[]): number {
    let conflicts = 0;

    for (const fact of newFacts) {
      const existing = this.findConflict(fact);
      if (existing) {
        // Supersede the old fact
        existing.validUntil = fact.validFrom;
        conflicts++;
      }
      this.indexFact(fact);
      this.dirty = true;
    }

    return conflicts;
  }

  /** Find an existing fact that conflicts with the new one */
  private findConflict(newFact: ExtractedFact): ExtractedFact | null {
    const entityFacts = this.byEntity.get(newFact.entityNorm);
    if (!entityFacts) return null;

    for (const id of entityFacts) {
      const existing = this.facts.get(id);
      if (!existing || existing.validUntil) continue; // Already superseded

      // Same entity + same relation = conflict
      if (existing.relation === newFact.relation) {
        // Check if values are similar enough to be an update
        if (this.isSimilarValue(existing.value, newFact.value)) {
          return existing;
        }
        // Different values for same relation = supersede old
        // e.g., "lives_in NYC" superseded by "lives_in London"
        if (this.isExclusiveRelation(newFact.relation)) {
          return existing;
        }
      }
    }

    return null;
  }

  /** Check if two values are similar enough to be considered the same fact */
  private isSimilarValue(a: string, b: string): boolean {
    const na = a.toLowerCase().trim();
    const nb = b.toLowerCase().trim();
    if (na === nb) return true;
    // One contains the other
    if (na.includes(nb) || nb.includes(na)) return true;
    return false;
  }

  /** Relations where only one value can be active at a time */
  private isExclusiveRelation(relation: string): boolean {
    const exclusive = new Set([
      'lives_in', 'works_at', 'age', 'favorite_color', 'favorite_food',
      'favorite_restaurant', 'favorite_movie', 'favorite_book',
      'favorite_music', 'favorite_sport', 'job_title', 'employer',
      'marital_status', 'city', 'country'
    ]);
    return exclusive.has(relation);
  }

  /** Index a fact in all lookup maps */
  private indexFact(fact: ExtractedFact): void {
    this.facts.set(fact.id, fact);

    if (!this.byEntity.has(fact.entityNorm)) {
      this.byEntity.set(fact.entityNorm, new Set());
    }
    this.byEntity.get(fact.entityNorm)!.add(fact.id);

    if (!this.byRelation.has(fact.relation)) {
      this.byRelation.set(fact.relation, new Set());
    }
    this.byRelation.get(fact.relation)!.add(fact.id);

    if (!this.byCategory.has(fact.category)) {
      this.byCategory.set(fact.category, new Set());
    }
    this.byCategory.get(fact.category)!.add(fact.id);
  }

  /** Save facts to disk (full rewrite for consistency) */
  save(): void {
    if (!this.dirty && fs.existsSync(this.factsPath)) return;

    const dir = path.dirname(this.factsPath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }

    const lines = Array.from(this.facts.values())
      .map(f => JSON.stringify(f))
      .join('\n');
    fs.writeFileSync(this.factsPath, lines + '\n', 'utf-8');
    this.dirty = false;
  }

  /** Append new facts to disk (faster than full rewrite) */
  append(facts: ExtractedFact[]): void {
    const dir = path.dirname(this.factsPath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }

    const lines = facts.map(f => JSON.stringify(f)).join('\n');
    fs.appendFileSync(this.factsPath, lines + '\n', 'utf-8');
  }

  // ─── Query methods ──────────────────────────────────────────────────────

  /** Get all active facts for an entity */
  getEntityFacts(entity: string): ExtractedFact[] {
    const norm = normalizeEntity(entity);
    const ids = this.byEntity.get(norm);
    if (!ids) return [];
    return Array.from(ids)
      .map(id => this.facts.get(id)!)
      .filter(f => f && !f.validUntil);
  }

  /** Get all active facts for a relation */
  getRelationFacts(relation: string): ExtractedFact[] {
    const ids = this.byRelation.get(relation);
    if (!ids) return [];
    return Array.from(ids)
      .map(id => this.facts.get(id)!)
      .filter(f => f && !f.validUntil);
  }

  /** Get all active facts in a category */
  getCategoryFacts(category: string): ExtractedFact[] {
    const ids = this.byCategory.get(category);
    if (!ids) return [];
    return Array.from(ids)
      .map(id => this.facts.get(id)!)
      .filter(f => f && !f.validUntil);
  }

  /** Get all active preferences */
  getPreferences(): ExtractedFact[] {
    return this.getCategoryFacts('preference');
  }

  /** Search facts by text query (simple keyword match) */
  searchFacts(query: string): ExtractedFact[] {
    const terms = query.toLowerCase().split(/\s+/);
    const results: ExtractedFact[] = [];

    for (const fact of this.facts.values()) {
      if (fact.validUntil) continue; // Skip superseded

      const text = `${fact.entity} ${fact.relation} ${fact.value} ${fact.rawText}`.toLowerCase();
      const matches = terms.filter(t => text.includes(t)).length;
      if (matches >= Math.ceil(terms.length * 0.5)) {
        results.push(fact);
      }
    }

    return results;
  }

  /** Get facts valid at a specific time */
  getFactsAt(timestamp: string): ExtractedFact[] {
    const t = new Date(timestamp).getTime();
    const results: ExtractedFact[] = [];

    for (const fact of this.facts.values()) {
      const from = new Date(fact.validFrom).getTime();
      const until = fact.validUntil ? new Date(fact.validUntil).getTime() : Infinity;
      if (t >= from && t < until) {
        results.push(fact);
      }
    }

    return results;
  }

  /** Get stats */
  stats(): FactStoreStats {
    const active = Array.from(this.facts.values()).filter(f => !f.validUntil);
    return {
      totalFacts: this.facts.size,
      activeFacts: active.length,
      supersededFacts: this.facts.size - active.length,
      entities: this.byEntity.size,
      relations: this.byRelation.size
    };
  }

  /** Get all facts (for testing/debugging) */
  getAllFacts(): ExtractedFact[] {
    return Array.from(this.facts.values());
  }
}
