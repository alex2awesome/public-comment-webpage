import { FeedbackRequest, RolloutResult } from "../types";

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
