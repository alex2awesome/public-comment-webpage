export interface BibliographyEntry {
  paperId?: string;
  title?: string;
  year?: number;
  url?: string;
  reason?: string;
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
  [key: string]: unknown;
}

export type FeedbackSentiment = "positive" | "negative";

export interface FeedbackRequest {
  runId: string;
  langsmithRunId?: string;
  sentiment: FeedbackSentiment;
  note?: string;
}
