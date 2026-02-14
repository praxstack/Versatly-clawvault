import * as fs from 'fs';
import * as path from 'path';
import { listTasks, type Task } from './task-utils.js';
import { readObservations, parseObservationLines } from './observation-reader.js';
import {
  type Canvas,
  type CanvasNode,
  createTextNode,
  createFileNode,
  createGroupWithNodes,
  flattenGroups,
  getPriorityColor,
  CANVAS_COLORS,
  LAYOUT,
  type GroupWithNodes
} from './canvas-layout.js';
import type { CanvasTemplate, CanvasTemplateOptions } from './canvas-templates.js';

const DAY_MS = 24 * 60 * 60 * 1000;
const RECENT_DAYS = 7;
const DEFAULT_CANVAS_WIDTH = 1120;
const GROUP_GAP = 40;
const MAX_DECISIONS = 12;
const MAX_OPEN_LOOPS = 12;

function listRecentDecisionPaths(vaultPath: string, days: number): string[] {
  const decisionsDir = path.join(vaultPath, 'decisions');
  if (!fs.existsSync(decisionsDir)) {
    return [];
  }

  const cutoff = Date.now() - (days * DAY_MS);
  return fs.readdirSync(decisionsDir, { withFileTypes: true })
    .filter((entry) => entry.isFile() && entry.name.endsWith('.md'))
    .map((entry) => {
      const absolutePath = path.join(decisionsDir, entry.name);
      const mtimeMs = fs.statSync(absolutePath).mtimeMs;
      return {
        relativePath: `decisions/${entry.name}`,
        mtimeMs
      };
    })
    .filter((entry) => entry.mtimeMs >= cutoff)
    .sort((left, right) => right.mtimeMs - left.mtimeMs)
    .slice(0, MAX_DECISIONS)
    .map((entry) => entry.relativePath);
}

function daysOpen(task: Task): number {
  const createdMs = new Date(task.frontmatter.created).getTime();
  if (!Number.isFinite(createdMs) || createdMs <= 0) {
    return 0;
  }
  return Math.floor((Date.now() - createdMs) / DAY_MS);
}

function summarizeObservations(vaultPath: string): string {
  const recentObservations = readObservations(vaultPath, RECENT_DAYS);
  if (!recentObservations.trim()) {
    return '**Recent Observations**\n\nNo observations in the last 7 days.';
  }

  const parsed = parseObservationLines(recentObservations);
  if (parsed.length === 0) {
    return '**Recent Observations**\n\nNo parseable observation entries.';
  }

  const byType = new Map<string, number>();
  for (const line of parsed) {
    byType.set(line.type, (byType.get(line.type) ?? 0) + 1);
  }

  const topInsights = [...parsed]
    .sort((left, right) => right.importance - left.importance)
    .slice(0, 3)
    .map((line) => `- ${line.content}`);

  const typeSummary = [...byType.entries()]
    .sort((left, right) => right[1] - left[1])
    .map(([type, count]) => `${type}: ${count}`)
    .join(' | ');

  return [
    '**Recent Observations**',
    '',
    `Entries (7d): ${parsed.length}`,
    `Breakdown: ${typeSummary || 'n/a'}`,
    '',
    '**Top signals**',
    ...(topInsights.length > 0 ? topInsights : ['- None'])
  ].join('\n');
}

function canvasWidthFromOptions(options: CanvasTemplateOptions): number {
  if (options.width && Number.isFinite(options.width)) {
    return Math.max(900, Math.floor(options.width));
  }
  return DEFAULT_CANVAS_WIDTH;
}

export function generateSprintCanvas(
  vaultPath: string,
  options: CanvasTemplateOptions = {}
): Canvas {
  const resolvedPath = path.resolve(vaultPath);
  const width = canvasWidthFromOptions(options);
  const tasks = listTasks(
    resolvedPath,
    options.project ? { project: options.project } : undefined
  );

  const activeTasks = tasks.filter((task) =>
    task.frontmatter.status === 'open' || task.frontmatter.status === 'in-progress'
  );
  const blockedTasks = tasks.filter((task) => task.frontmatter.status === 'blocked');
  const completedTasks = tasks.filter((task) => task.frontmatter.status === 'done');
  const completionRate = tasks.length > 0 ? Math.round((completedTasks.length / tasks.length) * 100) : 0;
  const staleOpenTasks = tasks
    .filter((task) => task.frontmatter.status !== 'done' && daysOpen(task) > RECENT_DAYS)
    .sort((left, right) => daysOpen(right) - daysOpen(left))
    .slice(0, MAX_OPEN_LOOPS);
  const recentDecisionPaths = listRecentDecisionPaths(resolvedPath, RECENT_DAYS);

  const metricNodeWidth = 220;
  const metricNodeHeight = LAYOUT.DEFAULT_NODE_HEIGHT + 10;
  const metricGap = 20;
  const totalMetricWidth = (metricNodeWidth * 3) + (metricGap * 2);
  const metricStartX = Math.max(0, Math.floor((width - totalMetricWidth) / 2));

  const nodes: CanvasNode[] = [
    createTextNode(
      metricStartX,
      0,
      metricNodeWidth,
      metricNodeHeight,
      `**Active Tasks**\n\n${activeTasks.length}`,
      CANVAS_COLORS.CYAN
    ),
    createTextNode(
      metricStartX + metricNodeWidth + metricGap,
      0,
      metricNodeWidth,
      metricNodeHeight,
      `**Blocked Tasks**\n\n${blockedTasks.length}`,
      CANVAS_COLORS.RED
    ),
    createTextNode(
      metricStartX + ((metricNodeWidth + metricGap) * 2),
      0,
      metricNodeWidth,
      metricNodeHeight,
      `**Completion**\n\n${completionRate}%`,
      CANVAS_COLORS.GREEN
    )
  ];

  const topSectionY = metricNodeHeight + 40;
  const columnWidth = Math.floor((width - GROUP_GAP) / 2);
  const fileNodeWidth = Math.max(180, columnWidth - (LAYOUT.GROUP_PADDING * 2));

  const decisionNodes: CanvasNode[] = [];
  if (recentDecisionPaths.length > 0) {
    for (const relativePath of recentDecisionPaths) {
      decisionNodes.push(createFileNode(0, 0, fileNodeWidth, LAYOUT.FILE_NODE_HEIGHT, relativePath, CANVAS_COLORS.PURPLE));
    }
  } else {
    decisionNodes.push(
      createTextNode(0, 0, fileNodeWidth, LAYOUT.SMALL_NODE_HEIGHT + 10, '_No recent decisions_')
    );
  }

  const openLoopNodes: CanvasNode[] = [];
  if (staleOpenTasks.length > 0) {
    for (const task of staleOpenTasks) {
      openLoopNodes.push(
        createFileNode(
          0,
          0,
          fileNodeWidth,
          LAYOUT.FILE_NODE_HEIGHT,
          `tasks/${task.slug}.md`,
          getPriorityColor(task.frontmatter.priority) ?? CANVAS_COLORS.YELLOW
        )
      );
    }
  } else {
    openLoopNodes.push(
      createTextNode(0, 0, fileNodeWidth, LAYOUT.SMALL_NODE_HEIGHT + 10, '_No open loops older than 7 days_')
    );
  }

  const topGroups: GroupWithNodes[] = [
    createGroupWithNodes(
      0,
      topSectionY,
      columnWidth,
      `Decisions (last ${RECENT_DAYS} days)`,
      decisionNodes,
      CANVAS_COLORS.PURPLE
    ),
    createGroupWithNodes(
      columnWidth + GROUP_GAP,
      topSectionY,
      columnWidth,
      'Open Loops (>7 days)',
      openLoopNodes,
      CANVAS_COLORS.YELLOW
    )
  ];
  nodes.push(...flattenGroups(topGroups));

  const bottomY = Math.max(
    ...topGroups.map((group) => group.group.y + group.group.height)
  ) + LAYOUT.GROUP_SPACING;
  const observationSummaryNode = createTextNode(
    0,
    0,
    width - (LAYOUT.GROUP_PADDING * 2),
    LAYOUT.DEFAULT_NODE_HEIGHT + 80,
    summarizeObservations(resolvedPath),
    CANVAS_COLORS.CYAN
  );
  const observationGroup = createGroupWithNodes(
    0,
    bottomY,
    width,
    options.project ? `Observation Summary (${options.project})` : 'Observation Summary',
    [observationSummaryNode],
    CANVAS_COLORS.CYAN
  );
  nodes.push(...flattenGroups([observationGroup]));

  return {
    nodes,
    edges: []
  };
}

export const sprintCanvasTemplate: CanvasTemplate = {
  id: 'sprint',
  name: 'Sprint Focus',
  description: 'Weekly focus board with sprint metrics, decisions, and open loops.',
  generate(vaultPath: string, options: CanvasTemplateOptions): Canvas {
    return generateSprintCanvas(vaultPath, options);
  }
};
