import * as path from 'path';
import { listTasks, type Task, type TaskStatus } from './task-utils.js';
import {
  type Canvas,
  type CanvasNode,
  type CanvasEdge,
  createFileNode,
  createTextNode,
  createGroupWithNodes,
  flattenGroups,
  createEdge,
  getPriorityColor,
  CANVAS_COLORS,
  LAYOUT,
  type GroupWithNodes
} from './canvas-layout.js';
import type { CanvasTemplate, CanvasTemplateOptions } from './canvas-templates.js';

interface BoardColumn {
  status: TaskStatus;
  label: string;
  groupColor?: string;
  fallbackFileColor?: string;
}

const COLUMN_DEFINITIONS: BoardColumn[] = [
  { status: 'open', label: 'Open' },
  { status: 'in-progress', label: 'In Progress', groupColor: CANVAS_COLORS.CYAN },
  { status: 'blocked', label: 'Blocked', groupColor: CANVAS_COLORS.RED, fallbackFileColor: CANVAS_COLORS.RED },
  { status: 'done', label: 'Done', groupColor: CANVAS_COLORS.GREEN, fallbackFileColor: CANVAS_COLORS.GREEN }
];

const DEFAULT_CANVAS_WIDTH = 1280;
const COLUMN_GAP = 35;

function toBlockedBySlug(value: string): string {
  return value
    .toLowerCase()
    .replace(/[^\w\s-]/g, '')
    .replace(/\s+/g, '-');
}

function getColumnWidth(totalWidth: number): number {
  const totalGap = COLUMN_GAP * (COLUMN_DEFINITIONS.length - 1);
  const available = Math.max(880, totalWidth) - totalGap;
  return Math.floor(available / COLUMN_DEFINITIONS.length);
}

function buildBlockedEdges(boardTasks: Task[], nodes: CanvasNode[]): CanvasEdge[] {
  const nodeByFile = new Map<string, CanvasNode>();
  for (const node of nodes) {
    if (node.type === 'file' && node.file) {
      nodeByFile.set(node.file, node);
    }
  }

  const taskSlugSet = new Set(boardTasks.map((task) => task.slug));
  const edges: CanvasEdge[] = [];

  for (const task of boardTasks) {
    if (task.frontmatter.status !== 'blocked' || !task.frontmatter.blocked_by) {
      continue;
    }
    const blockerSlug = toBlockedBySlug(task.frontmatter.blocked_by);
    if (!taskSlugSet.has(blockerSlug)) {
      continue;
    }

    const blockedNode = nodeByFile.get(`tasks/${task.slug}.md`);
    const blockerNode = nodeByFile.get(`tasks/${blockerSlug}.md`);
    if (blockedNode && blockerNode) {
      edges.push(createEdge(blockedNode.id, 'right', blockerNode.id, 'left', 'blocked by', CANVAS_COLORS.RED));
    }
  }

  return edges;
}

export function generateProjectBoardCanvas(
  vaultPath: string,
  options: CanvasTemplateOptions = {}
): Canvas {
  const resolvedPath = path.resolve(vaultPath);
  const tasks = listTasks(
    resolvedPath,
    options.project ? { project: options.project } : undefined
  );

  const tasksByStatus = new Map<TaskStatus, Task[]>();
  for (const column of COLUMN_DEFINITIONS) {
    tasksByStatus.set(column.status, []);
  }
  for (const task of tasks) {
    const bucket = tasksByStatus.get(task.frontmatter.status);
    if (bucket) {
      bucket.push(task);
    }
  }

  const boardWidth = options.width && Number.isFinite(options.width)
    ? Math.floor(options.width)
    : DEFAULT_CANVAS_WIDTH;
  const columnWidth = getColumnWidth(boardWidth);
  const fileNodeWidth = Math.max(180, columnWidth - (LAYOUT.GROUP_PADDING * 2));

  const boardGroups: GroupWithNodes[] = COLUMN_DEFINITIONS.map((column, index) => {
    const columnTasks = tasksByStatus.get(column.status) ?? [];
    const childNodes: CanvasNode[] = [];

    for (const task of columnTasks) {
      const color = getPriorityColor(task.frontmatter.priority) ?? column.fallbackFileColor;
      childNodes.push(
        createFileNode(
          0,
          0,
          fileNodeWidth,
          LAYOUT.FILE_NODE_HEIGHT,
          `tasks/${task.slug}.md`,
          color
        )
      );
    }

    if (columnTasks.length === 0) {
      childNodes.push(
        createTextNode(
          0,
          0,
          fileNodeWidth,
          LAYOUT.SMALL_NODE_HEIGHT + 10,
          '_No tasks_'
        )
      );
    }

    return createGroupWithNodes(
      index * (columnWidth + COLUMN_GAP),
      0,
      columnWidth,
      `${column.label} (${columnTasks.length})`,
      childNodes,
      column.groupColor
    );
  });

  const nodes = flattenGroups(boardGroups);
  const edges = buildBlockedEdges(tasks, nodes);
  const scopeNode = createTextNode(
    0,
    -(LAYOUT.SMALL_NODE_HEIGHT + 30),
    Math.min(boardWidth, 460),
    LAYOUT.SMALL_NODE_HEIGHT + 20,
    options.project
      ? `**Project Board**\nProject: ${options.project}\nTasks: ${tasks.length}`
      : `**Project Board**\nAll projects\nTasks: ${tasks.length}`,
    CANVAS_COLORS.CYAN
  );
  nodes.push(scopeNode);

  return { nodes, edges };
}

export const projectBoardCanvasTemplate: CanvasTemplate = {
  id: 'project-board',
  name: 'Project Board',
  description: 'Kanban-style board for tasks grouped by status.',
  generate(vaultPath: string, options: CanvasTemplateOptions): Canvas {
    return generateProjectBoardCanvas(vaultPath, options);
  }
};
