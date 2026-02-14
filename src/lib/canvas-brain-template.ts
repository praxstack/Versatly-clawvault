import * as path from 'path';
import { collectVaultStats } from './vault-stats.js';
import { loadMemoryGraphIndex, type MemoryGraphNode } from './memory-graph.js';
import {
  type Canvas,
  type CanvasNode,
  type CanvasEdge,
  createTextNode,
  createFileNode,
  createGroupWithNodes,
  createEdge,
  flattenGroups,
  CANVAS_COLORS,
  LAYOUT,
  type GroupWithNodes
} from './canvas-layout.js';
import type { CanvasTemplate, CanvasTemplateOptions } from './canvas-templates.js';

interface CategorySummary {
  category: string;
  fileCount: number;
  entities: MemoryGraphNode[];
}

const MAX_CATEGORY_GROUPS = 6;
const MAX_CATEGORY_ENTITIES = 5;
const DEFAULT_CANVAS_WIDTH = 1400;
const DEFAULT_CANVAS_HEIGHT = 1000;
const RADIAL_GROUP_WIDTH = 320;
const RADIAL_GROUP_MIN_Y = -80;
const RADIAL_GROUP_MAX_Y = 1200;

function toCategoryLabel(raw: string): string {
  if (!raw) {
    return 'uncategorized';
  }
  return raw.replace(/[-_]+/g, ' ').trim();
}

function categoryFromNode(node: MemoryGraphNode): string {
  if (node.path) {
    const firstSegment = node.path.split('/')[0]?.trim();
    if (firstSegment) {
      return firstSegment.toLowerCase();
    }
  }
  return node.category.toLowerCase();
}

function summarizeCategories(nodes: MemoryGraphNode[]): CategorySummary[] {
  const byCategory = new Map<string, MemoryGraphNode[]>();
  for (const node of nodes) {
    if (!node.path || node.type === 'tag' || node.type === 'unresolved') {
      continue;
    }
    const key = categoryFromNode(node);
    const bucket = byCategory.get(key) ?? [];
    bucket.push(node);
    byCategory.set(key, bucket);
  }

  return [...byCategory.entries()]
    .map(([category, entities]) => ({
      category,
      fileCount: entities.length,
      entities: entities
        .sort((left, right) => right.degree - left.degree)
        .slice(0, MAX_CATEGORY_ENTITIES)
    }))
    .sort((left, right) => right.fileCount - left.fileCount || left.category.localeCompare(right.category))
    .slice(0, MAX_CATEGORY_GROUPS);
}

function fallbackCategorySummaries(vaultPath: string): CategorySummary[] {
  const stats = collectVaultStats(vaultPath);
  return Object.entries(stats.documents.byCategory)
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .slice(0, MAX_CATEGORY_GROUPS)
    .map(([category, fileCount]) => ({
      category,
      fileCount,
      entities: []
    }));
}

function createCategoryColor(index: number): string {
  const palette = [
    CANVAS_COLORS.CYAN,
    CANVAS_COLORS.PURPLE,
    CANVAS_COLORS.YELLOW,
    CANVAS_COLORS.ORANGE,
    CANVAS_COLORS.GREEN
  ];
  return palette[index % palette.length];
}

function getCanvasSize(options: CanvasTemplateOptions): { width: number; height: number } {
  const width = options.width && Number.isFinite(options.width)
    ? Math.max(960, Math.floor(options.width))
    : DEFAULT_CANVAS_WIDTH;
  const height = options.height && Number.isFinite(options.height)
    ? Math.max(680, Math.floor(options.height))
    : DEFAULT_CANVAS_HEIGHT;
  return { width, height };
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export function generateBrainCanvas(
  vaultPath: string,
  options: CanvasTemplateOptions = {}
): Canvas {
  const resolvedPath = path.resolve(vaultPath);
  const memoryGraph = loadMemoryGraphIndex(resolvedPath)?.graph;
  const vaultName = path.basename(resolvedPath);
  const vaultStats = collectVaultStats(resolvedPath);
  const { width, height } = getCanvasSize(options);

  const categories = memoryGraph
    ? summarizeCategories(memoryGraph.nodes)
    : fallbackCategorySummaries(resolvedPath);
  const selectedCategories = categories.length > 0
    ? categories
    : [{
      category: 'vault',
      fileCount: vaultStats.documents.total,
      entities: []
    }];

  const centerGroupWidth = 360;
  const centerGroupX = Math.floor((width - centerGroupWidth) / 2);
  const centerGroupY = Math.floor((height - 220) / 2);
  const centerText = [
    `**${vaultName}**`,
    '',
    `Known files: ${memoryGraph?.stats.nodeCount ?? 0}`,
    `Wiki links: ${memoryGraph?.stats.edgeCount ?? 0}`,
    `Tasks: ${vaultStats.tasks.total}`,
    `Observations: ${vaultStats.observations.total}`,
    memoryGraph ? '' : '_Graph index unavailable_'
  ].join('\n');

  const centerGroup = createGroupWithNodes(
    centerGroupX,
    centerGroupY,
    centerGroupWidth,
    'Brain Core',
    [
      createTextNode(
        0,
        0,
        centerGroupWidth - (LAYOUT.GROUP_PADDING * 2),
        LAYOUT.DEFAULT_NODE_HEIGHT + 30,
        centerText,
        CANVAS_COLORS.CYAN
      )
    ],
    CANVAS_COLORS.PURPLE
  );

  const radialGroups: GroupWithNodes[] = [];
  const entityNodeByGraphId = new Map<string, CanvasNode>();
  const edges: CanvasEdge[] = [];

  const centerPointX = centerGroup.group.x + (centerGroup.group.width / 2);
  const centerPointY = centerGroup.group.y + (centerGroup.group.height / 2);
  const radialDistance = Math.max(300, Math.floor(Math.min(width, height) * 0.34));
  const nodeWidth = RADIAL_GROUP_WIDTH - (LAYOUT.GROUP_PADDING * 2);

  selectedCategories.forEach((category, index) => {
    const angle = (Math.PI * 2 * index) / selectedCategories.length;
    const targetX = Math.round(centerPointX + (Math.cos(angle) * radialDistance)) - Math.floor(RADIAL_GROUP_WIDTH / 2);
    const targetY = clamp(
      Math.round(centerPointY + (Math.sin(angle) * radialDistance)) - 140,
      RADIAL_GROUP_MIN_Y,
      RADIAL_GROUP_MAX_Y
    );

    const childNodes: CanvasNode[] = [
      createTextNode(
        0,
        0,
        nodeWidth,
        LAYOUT.SMALL_NODE_HEIGHT + 10,
        `Files: ${category.fileCount}`
      )
    ];

    const entityNodeIds: Array<{ graphId: string; canvasId: string }> = [];
    for (const entity of category.entities) {
      if (!entity.path) {
        continue;
      }
      const fileNode = createFileNode(
        0,
        0,
        nodeWidth,
        LAYOUT.FILE_NODE_HEIGHT,
        entity.path
      );
      entityNodeIds.push({ graphId: entity.id, canvasId: fileNode.id });
      childNodes.push(fileNode);
    }

    if (category.entities.length === 0) {
      childNodes.push(
        createTextNode(
          0,
          0,
          nodeWidth,
          LAYOUT.SMALL_NODE_HEIGHT + 10,
          '_No linked entities_'
        )
      );
    }

    const categoryGroup = createGroupWithNodes(
      targetX,
      targetY,
      RADIAL_GROUP_WIDTH,
      `${toCategoryLabel(category.category)} (${category.fileCount})`,
      childNodes,
      createCategoryColor(index)
    );
    radialGroups.push(categoryGroup);

    for (const mapping of entityNodeIds) {
      const canvasNode = categoryGroup.nodes.find((node) => node.id === mapping.canvasId);
      if (canvasNode) {
        entityNodeByGraphId.set(mapping.graphId, canvasNode);
      }
    }

    const sourceOnRight = categoryGroup.group.x >= centerGroup.group.x;
    edges.push(
      createEdge(
        centerGroup.group.id,
        sourceOnRight ? 'right' : 'left',
        categoryGroup.group.id,
        sourceOnRight ? 'left' : 'right',
        'category',
        CANVAS_COLORS.CYAN
      )
    );
  });

  if (memoryGraph) {
    const seenEdgePairs = new Set<string>();
    for (const graphEdge of memoryGraph.edges) {
      if (graphEdge.type !== 'wiki_link') {
        continue;
      }
      const sourceNode = entityNodeByGraphId.get(graphEdge.source);
      const targetNode = entityNodeByGraphId.get(graphEdge.target);
      if (!sourceNode || !targetNode || sourceNode.id === targetNode.id) {
        continue;
      }

      const pairKey = [sourceNode.id, targetNode.id].sort((left, right) => left.localeCompare(right)).join('::');
      if (seenEdgePairs.has(pairKey)) {
        continue;
      }
      seenEdgePairs.add(pairKey);
      edges.push(createEdge(sourceNode.id, 'right', targetNode.id, 'left', 'wiki-link', CANVAS_COLORS.PURPLE));
    }
  }

  const nodes = [...flattenGroups([centerGroup]), ...flattenGroups(radialGroups)];
  return { nodes, edges };
}

export const brainCanvasTemplate: CanvasTemplate = {
  id: 'brain',
  name: 'Brain Overview',
  description: 'Radial knowledge map with category hubs and linked entities.',
  generate(vaultPath: string, options: CanvasTemplateOptions): Canvas {
    return generateBrainCanvas(vaultPath, options);
  }
};
