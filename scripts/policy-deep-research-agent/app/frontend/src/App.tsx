import { useEffect, useRef, useState } from "react";
import QueryForm from "./components/QueryForm";
import RunStatus from "./components/RunStatus";
import FeedbackPanel, { FeedbackFormValues } from "./components/FeedbackPanel";
import { API_BASE_URL, normalizeEpisode, normalizeSummary, regenerateMemo, sendFeedback } from "./lib/api";
import { FindingsSummary, RolloutResult, RolloutStreamEvent } from "./types";
import "./App.css";

const BACKEND_WAKE_MESSAGES = [
  "Warming up the policy research agent…",
  "Coaxing the backend awake. This usually takes a few seconds.",
  "Still stretching… prepping LangGraph tools.",
  "Almost there! Checking cache and LangSmith telemetry.",
  "Thanks for your patience—we're pinging the backend again.",
];

export type RunState = "idle" | "running" | "error" | "complete";
type FeedbackState = "idle" | "sending" | "sent" | "error";

function App() {
  const [question, setQuestion] = useState("");
  const [maxSteps, setMaxSteps] = useState(4);
  const enableBibliography = true;
  const [runState, setRunState] = useState<RunState>("idle");
  const [result, setResult] = useState<RolloutResult | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [feedbackState, setFeedbackState] = useState<FeedbackState>("idle");
  const [feedbackError, setFeedbackError] = useState<string | null>(null);
  const [events, setEvents] = useState<RolloutStreamEvent[]>([]);
  const [showEvents, setShowEvents] = useState(true);
  const [backendReady, setBackendReady] = useState(false);
  const [backendStatusMessage, setBackendStatusMessage] = useState(BACKEND_WAKE_MESSAGES[0]);
  const [latestSummary, setLatestSummary] = useState<FindingsSummary | null>(null);
  const [memoUpdateStatus, setMemoUpdateStatus] = useState<"idle" | "saving" | "success" | "error">("idle");
  const [memoUpdateError, setMemoUpdateError] = useState<string | null>(null);
  const [memoHistory, setMemoHistory] = useState<string[]>([]);
  const [summaryHistory, setSummaryHistory] = useState<(FindingsSummary | null)[]>([]);
  const [revisionIndex, setRevisionIndex] = useState(0);
  const eventSourceRef = useRef<EventSource | null>(null);
  const warmupTimerRef = useRef<number | null>(null);

  const closeEventStream = () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
  };

  useEffect(() => {
    let cancelled = false;
    let attempt = 0;
    const pollBackend = async () => {
      attempt += 1;
      setBackendStatusMessage(BACKEND_WAKE_MESSAGES[(attempt - 1) % BACKEND_WAKE_MESSAGES.length]);
      try {
        const response = await fetch(`${API_BASE_URL}/healthz`, {
          method: "GET",
          cache: "no-store",
        });
        if (!response.ok) {
          throw new Error(`Healthcheck failed: ${response.status}`);
        }
        if (!cancelled) {
          setBackendReady(true);
          setBackendStatusMessage("");
        }
      } catch (err) {
        if (cancelled) {
          return;
        }
        const nextRun = window.setTimeout(pollBackend, 1000);
        warmupTimerRef.current = nextRun;
      }
    };
    pollBackend();
    return () => {
      cancelled = true;
      closeEventStream();
      if (warmupTimerRef.current) {
        window.clearTimeout(warmupTimerRef.current);
      }
    };
  }, []);

  const attachEventHandler = (type: string, data: string) => {
    let payload: RolloutStreamEvent;
    try {
      payload = JSON.parse(data);
    } catch (err) {
      payload = { type, message: data };
    }
    if (!payload.type) {
      payload.type = type;
    }
    if (payload.type === "tool_result" && payload.tool === "summarize_findings" && payload.result?.summary) {
      const summary = normalizeSummary(payload.result.summary);
      if (summary) {
        setLatestSummary(summary);
      }
    }
    if (payload.type === "complete" || payload.type === "run_completed") {
      if (payload.episode) {
        const normalized = normalizeEpisode(payload.episode);
        payload.episode = normalized;
        setResult(normalized);
        if (normalized.summary) {
          setLatestSummary(normalized.summary);
        }
        setSummaryHistory([normalized.summary ?? null]);
        const memoText = normalized.finalMemo || "";
        if (memoText) {
          setMemoHistory([memoText]);
          setRevisionIndex(0);
        } else {
          setMemoHistory([]);
          setRevisionIndex(0);
        }
      }
      setRunState("complete");
      setShowEvents(false);
      closeEventStream();
    } else if (payload.type === "error") {
      setErrorMessage(payload.message || "Rollout failed.");
      setRunState("error");
      closeEventStream();
    } else {
      setRunState("running");
      setErrorMessage(null);
    }
    setEvents((prev) => [...prev, payload]);
  };

  const handleToggleEvents = () => {
    if (events.length === 0) {
      return;
    }
    setShowEvents((prev) => !prev);
  };

  const handleStartQuery = () => {
    if (!backendReady) {
      setErrorMessage(
        "Still waking the backend up—give it just another moment! (This can take 20-30 seconds because we're cheap and are hosting this on free architecture.)"
      );
      return;
    }
    if (!question.trim()) {
      setErrorMessage("Enter a policy research question to start the agent.");
      setRunState("error");
      return;
    }

    closeEventStream();
    setRunState("running");
    setErrorMessage(null);
    setResult(null);
    setFeedbackState("idle");
    setFeedbackError(null);
    setEvents([]);
    setShowEvents(true);
    setLatestSummary(null);
    setMemoUpdateStatus("idle");
    setMemoUpdateError(null);
    setMemoHistory([]);
    setSummaryHistory([]);
    setRevisionIndex(0);

    try {
      const url = new URL("/rollouts/stream", API_BASE_URL);
      url.searchParams.set("question", question.trim());
      url.searchParams.set("max_steps", String(maxSteps));
      url.searchParams.set("enable_bibliography", enableBibliography ? "true" : "false");

      const source = new EventSource(url.toString());
      eventSourceRef.current = source;

      const defaultHandler = (event: MessageEvent<string>) => attachEventHandler(event.type || "message", event.data);
      source.onmessage = defaultHandler;
      const namedEvents = ["tool_call", "tool_result", "run_started", "run_completed", "complete", "error"];
      namedEvents.forEach((eventName) => {
        source.addEventListener(eventName, (event) => {
          const message = event as MessageEvent<string>;
          attachEventHandler(eventName, message.data);
        });
      });
      source.onerror = () => {
        // EventSource automatically retries transient failures; only surface an error if it closes.
        if (source.readyState === EventSource.CLOSED) {
          setRunState("error");
          setErrorMessage("Connection lost while streaming rollout.");
          closeEventStream();
        }
      };
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unable to start the rollout.";
      setErrorMessage(message);
      setRunState("error");
      closeEventStream();
    }
  };

  const handleFeedbackSubmit = async (values: FeedbackFormValues) => {
    if (!result) {
      return;
    }
    setFeedbackState("sending");
    setFeedbackError(null);
    try {
      await sendFeedback({
        runId: result.langsmithRunId ?? result.runId,
        langsmithRunId: result.langsmithRunId,
        sentiment: values.sentiment,
        note: values.note,
      });
      setFeedbackState("sent");
    } catch (err) {
      setFeedbackState("error");
      const message = err instanceof Error ? err.message : "Unable to send feedback.";
      setFeedbackError(message);
    }
  };

  const handleMemoRegenerate = async (editedSummary: FindingsSummary, directives: string) => {
    if (!result) {
      return;
    }
    const memoForContext = memoHistory[revisionIndex] ?? result.finalMemo ?? "";
    const toolEventsPayload = events
      .filter((event) => event.type === "tool_result" || event.type === "tool_call")
      .slice(-20)
      .map((event) => ({
        type: event.type,
        step: event.step,
        tool: event.tool,
        message: event.message,
        args: event.args,
        result: event.result,
      }));
    setMemoUpdateStatus("saving");
    setMemoUpdateError(null);
    try {
      const memo = await regenerateMemo({
        runId: result.runId,
        question: result.question,
        summary: editedSummary,
        notes: result.notes ?? [],
        directives,
        toolEvents: toolEventsPayload,
        priorMemo: memoForContext || undefined,
      });
      setResult((prev) => (prev ? { ...prev, finalMemo: memo } : prev));
      setLatestSummary(editedSummary);
      setSummaryHistory((prev) => {
        const next = [...prev, editedSummary];
        setRevisionIndex(next.length - 1);
        return next;
      });
      setMemoHistory((prev) => [...prev, memo]);
      setMemoUpdateStatus("success");
      window.setTimeout(() => setMemoUpdateStatus("idle"), 2500);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unable to update memo.";
      setMemoUpdateError(message);
      setMemoUpdateStatus("error");
    }
  };

  const handleRevisionNavigate = (direction: "prev" | "next") => {
    setRevisionIndex((prev) => {
      const maxIndex = Math.max(memoHistory.length, summaryHistory.length) - 1;
      const cappedMax = Math.max(maxIndex, 0);
      if (direction === "prev") {
        return Math.max(prev - 1, 0);
      }
      if (direction === "next") {
        return Math.min(prev + 1, cappedMax);
      }
      return prev;
    });
  };

  const revisionCount = Math.max(memoHistory.length, summaryHistory.length);
  const currentPlanSummary = summaryHistory[revisionIndex] ?? latestSummary;

  return (
    <div className={`app-shell${backendReady ? "" : " app-shell--waiting"}`}>
      {!backendReady && (
        <div className="startup-veil" role="status" aria-live="polite">
          <div className="startup-card">
            <div className="spinner" aria-hidden="true" />
            <div>
              <p className="eyebrow">Waking backend</p>
              <p className="startup-message">{backendStatusMessage}</p>
              <p className="startup-subtle">
                We’ll start streaming events the moment the API responds. (This can take 20-30 seconds because... we're
                cheap and are hosting this on free architecture...)
              </p>
            </div>
          </div>
        </div>
      )}
      <header className="app-hero">
        <div className="hero-card">
          <div className="hero-copy">
            <p className="eyebrow">Policy Research Agent</p>
            <h1>Policy Research Agent</h1>
            <p className="intro">
              A research agent that guides you through complex policy questions. 
              Review the synthesized memo before sharing structured feedback.
              Any thoughts on how to improve this process would be greatly appreciated!
            </p>
          </div>
        </div>
      </header>
      <main className="app-layout">
        <section className="surface surface-query">
          <div className="section-heading">
            <p className="section-eyebrow">Define your inquiry</p>
            <h2>Compose a research brief</h2>
            <p className="section-copy">Set question and guardrails for the agent to follow before it begins.</p>
          </div>
          <QueryForm
            question={question}
            maxSteps={maxSteps}
            disabled={runState === "running" || !backendReady}
            onQuestionChange={setQuestion}
            onMaxStepsChange={setMaxSteps}
            onSubmit={handleStartQuery}
          />
        </section>
        <section className="surface surface-output">
          <div className="section-heading">
            <p className="section-eyebrow">Live briefing</p>
            <h2>Research output</h2>
            <p className="section-copy">Tool calls in real time will show here, then final memo with citations will be displayed below.</p>
          </div>
          <RunStatus
            status={runState}
            result={result}
            error={errorMessage}
            events={events}
            showEvents={showEvents}
            maxSteps={maxSteps}
            onToggleEvents={handleToggleEvents}
            planSummary={currentPlanSummary ?? null}
            onSummaryUpdate={handleMemoRegenerate}
            memoUpdateStatus={memoUpdateStatus}
            memoUpdateError={memoUpdateError}
            memoHistory={memoHistory}
            revisionIndex={revisionIndex}
            revisionCount={revisionCount || (currentPlanSummary || memoHistory.length ? 1 : 0)}
            onRevisionNavigate={handleRevisionNavigate}
          />
          {runState === "complete" && result && (
            <div className="section-divider">
              <FeedbackPanel
                status={feedbackState}
                error={feedbackError}
                disabled={feedbackState === "sending"}
                onSubmit={handleFeedbackSubmit}
              />
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
