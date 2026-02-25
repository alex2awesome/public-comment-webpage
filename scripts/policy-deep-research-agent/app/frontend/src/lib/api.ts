import { FeedbackRequest, FindingsSummary, FindingsSummaryArticle, RegenerateMemoRequest, RolloutResult } from "../types";

export const API_BASE_URL = (import.meta.env.VITE_AGENT_API_URL ?? "http://localhost:8000").replace(/\/+$/, "");
const OPENAI_KEY = import.meta.env.VITE_OPENAI_API_KEY;

function buildHeaders(): HeadersInit {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (OPENAI_KEY) {
    headers["Authorization"] = `Bearer ${OPENAI_KEY}`;
  }
  return headers;
}

async function parseError(response: Response): Promise<string> {
  let message = `Request failed with status ${response.status}`;
  try {
    const payload = await response.json();
    if (typeof payload?.detail === "string") {
      message = payload.detail;
    } else if (typeof payload?.error === "string") {
      message = payload.error;
    }
  } catch (_) {
    // ignore JSON parse issues
  }
  return message;
}

export const normalizeEpisode = (payload: any): RolloutResult => {
  const summary = normalizeSummary(payload.summary ?? payload.findings_summary ?? null);
  const finalMemoBlocks = payload.final_memo_blocks ?? payload.finalMemoBlocks ?? null;
  const sourceDocuments = payload.source_documents ?? payload.sourceDocuments ?? [];
  return {
    runId: payload.run_id ?? payload.runId ?? "",
    taskId: payload.task_id ?? payload.taskId ?? "",
    question: payload.question ?? "",
    finalMemo: payload.final_memo ?? payload.finalMemo ?? "",
    bib: payload.bib ?? [],
    notes: payload.notes ?? [],
    steps: payload.steps ?? payload.step_count ?? 0,
    reward: payload.reward ?? 0,
    rewardBreakdown: payload.reward_breakdown ?? payload.rewardBreakdown ?? {},
    toolCalls: payload.tool_calls ?? payload.toolCalls ?? [],
    langsmithRunId: payload.langsmith_run_id ?? payload.langsmithRunId,
    summary,
    finalMemoBlocks,
    sourceDocuments,
  };
};

export async function sendFeedback(request: FeedbackRequest): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/feedback`, {
    method: "POST",
    headers: buildHeaders(),
    body: JSON.stringify({
      run_id: request.runId,
      langsmith_run_id: request.langsmithRunId,
      sentiment: request.sentiment,
      note: request.note,
    }),
  });

  if (!response.ok) {
    throw new Error(await parseError(response));
  }
}

export const normalizeSummary = (payload: any): FindingsSummary | null => {
  if (!payload) {
    return null;
  }
  const coerceList = (value: any): any[] => {
    if (Array.isArray(value)) {
      return value;
    }
    if (value === undefined || value === null) {
      return [];
    }
    return [value];
  };
  const normalizeArticle = (article: any): FindingsSummaryArticle | null => {
    if (!article || typeof article !== "object") {
      return null;
    }
    const authors = coerceList(article.authors ?? article.author_list).map((name) => String(name));
    const title = article.title ?? article.name ?? "";
    const paperId = article.paperId ?? article.paper_id ?? article.id;
    if (!title && !article.url && !paperId) {
      return null;
    }
    const fallbackUrl = paperId ? `https://www.semanticscholar.org/paper/${paperId}` : undefined;
    return {
      id: paperId ?? article.id,
      paperId,
      title,
      url: article.url ?? article.link ?? fallbackUrl,
      authors,
      reason_chosen: article.reason_chosen ?? article.reason ?? article.summary,
    };
  };
  const topArgsSource = payload["top arguments"] ?? payload.top_arguments ?? payload.topArguments ?? [];
  const topRecommendationsSource =
    payload["top recommendations"] ?? payload.top_recommendations ?? payload.topRecommendations ?? [];
  const topArticlesSource = payload["top articles"] ?? payload.top_articles ?? payload.topArticles ?? [];
  const topPeopleSource = payload["top people"] ?? payload.top_people ?? payload.topPeople ?? [];
  return {
    topArguments: coerceList(topArgsSource).map((arg) => String(arg)),
    topRecommendations: coerceList(topRecommendationsSource).map((rec) => String(rec)),
    topArticles: coerceList(topArticlesSource)
      .map(normalizeArticle)
      .filter((article): article is FindingsSummaryArticle => Boolean(article)),
    topPeople: coerceList(topPeopleSource).map((person) => String(person)),
  };
};

export async function regenerateMemo(request: RegenerateMemoRequest): Promise<string> {
  const payload = {
    run_id: request.runId,
    question: request.question,
    summary: {
      top_arguments: request.summary.topArguments,
      top_articles: request.summary.topArticles.map((article) => ({
        paperId: article.paperId ?? article.id,
        title: article.title,
        url: article.url,
        authors: article.authors ?? [],
        reason_chosen: article.reason_chosen,
      })),
      top_people: request.summary.topPeople,
      top_recommendations: request.summary.topRecommendations,
    },
    notes: request.notes,
    directives: request.directives,
    tool_events: request.toolEvents,
    prior_memo: request.priorMemo,
  };
  const response = await fetch(`${API_BASE_URL}/rollouts/resubmit`, {
    method: "POST",
    headers: buildHeaders(),
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(await parseError(response));
  }
  const data = await response.json();
  return data.memo;
}
