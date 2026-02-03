export interface BibliographyEntry {
  paperId?: string;
  title?: string;
  year?: number;
  url?: string;
  reason?: string;
}

export interface FindingsSummaryArticle {
  id?: string;
  paperId?: string;
  title?: string;
  url?: string;
  authors?: string[];
  reason_chosen?: string;
}

export interface FindingsSummary {
  topArguments: string[];
  topArticles: FindingsSummaryArticle[];
  topPeople: string[];
}

export interface ToolCallSummary {
  step: number;
  tool: string;
  args: Record<string, unknown>;
}

export interface RolloutResult {
  runId: string;
  taskId: string;
  question: string;
  finalMemo: string;
  bib: BibliographyEntry[];
  notes: string[];
  steps: number;
  reward: number;
  rewardBreakdown: Record<string, number>;
  toolCalls: ToolCallSummary[];
  langsmithRunId?: string;
  summary?: FindingsSummary | null;
}

export interface RolloutStreamEvent {
  type: string;
  step?: number;
  tool?: string;
  args?: Record<string, unknown>;
  result?: Record<string, unknown>;
  message?: string;
  question?: string;
  episode?: RolloutResult;
  summary?: FindingsSummary;
  [key: string]: unknown;
}

export type FeedbackSentiment = "positive" | "negative";

export interface FeedbackRequest {
  runId: string;
  langsmithRunId?: string;
  sentiment: FeedbackSentiment;
  note?: string;
}

export interface RegenerateMemoRequest {
  runId: string;
  question: string;
  summary: FindingsSummary;
  notes: string[];
  directives?: string;
  toolEvents: Record<string, unknown>[];
  priorMemo?: string;
}
