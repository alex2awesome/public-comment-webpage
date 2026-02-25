export interface BibliographyEntry {
  paperId?: string;
  title?: string;
  year?: number;
  url?: string;
  reason?: string;
  abstract?: string;
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
  topRecommendations: string[];
  topArticles: FindingsSummaryArticle[];
  topPeople: string[];
}

export interface ToolCallSummary {
  step: number;
  tool: string;
  args: Record<string, unknown>;
}

export interface MemoCitation {
  type?: string;
  document_index?: number;
  document_id?: string;
  document_title?: string;
  document_url?: string;
  paper_id?: string;
  cited_text?: string;
  start_char_index?: number;
  end_char_index?: number;
  start_page_number?: number;
  end_page_number?: number;
  start_block_index?: number;
  end_block_index?: number;
}

export interface MemoBlock {
  text: string;
  citations: MemoCitation[];
}

export interface SourceDocument {
  document_index: number;
  document_id?: string;
  paper_id?: string;
  title?: string;
  url?: string;
  year?: number;
  reason?: string;
  authors?: string;
  text_preview?: string;
  kind?: string;
  argument_index?: number;
  person_index?: number;
  recommendation_index?: number;
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
  finalMemoBlocks?: MemoBlock[] | null;
  sourceDocuments: SourceDocument[];
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
