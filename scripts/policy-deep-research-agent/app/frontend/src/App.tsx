import { useEffect, useRef, useState } from "react";
import QueryForm from "./components/QueryForm";
import RunStatus from "./components/RunStatus";
import FeedbackPanel, { FeedbackFormValues } from "./components/FeedbackPanel";
import { API_BASE_URL, normalizeEpisode, sendFeedback } from "./lib/api";
import { RolloutResult, RolloutStreamEvent } from "./types";
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
  const [maxSteps, setMaxSteps] = useState(10);
  const enableBibliography = false;
  const [runState, setRunState] = useState<RunState>("idle");
  const [result, setResult] = useState<RolloutResult | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [feedbackState, setFeedbackState] = useState<FeedbackState>("idle");
  const [feedbackError, setFeedbackError] = useState<string | null>(null);
  const [events, setEvents] = useState<RolloutStreamEvent[]>([]);
  const [showEvents, setShowEvents] = useState(true);
  const [backendReady, setBackendReady] = useState(false);
  const [backendStatusMessage, setBackendStatusMessage] = useState(BACKEND_WAKE_MESSAGES[0]);
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
    if (payload.type === "complete" || payload.type === "run_completed") {
      if (payload.episode) {
        const normalized = normalizeEpisode(payload.episode);
        payload.episode = normalized;
        setResult(normalized);
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
      setErrorMessage("Still waking the backend up—give it just another moment!");
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
        setRunState("error");
        setErrorMessage("Connection lost while streaming rollout.");
        closeEventStream();
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

  return (
    <div className={`app-shell${backendReady ? "" : " app-shell--waiting"}`}>
      {!backendReady && (
        <div className="startup-veil" role="status" aria-live="polite">
          <div className="startup-card">
            <div className="spinner" aria-hidden="true" />
            <div>
              <p className="eyebrow">Waking backend</p>
              <p className="startup-message">{backendStatusMessage}</p>
              <p className="startup-subtle">We’ll start streaming events the moment the API responds.</p>
            </div>
          </div>
        </div>
      )}
      <header className="app-header">
        <div>
          <p className="eyebrow">Policy Deep Research Agent</p>
          <h1>LangGraph Rollout Console</h1>
          <p className="intro">
            Submit a policy question, cap the number of LangGraph tool calls, and watch the research agent
            produce a memo. When you finish reviewing the answer, send structured feedback back to LangSmith.
          </p>
        </div>
      </header>
      <main className="app-layout">
        <section className="panel">
          <h2>Configure Query</h2>
          <QueryForm
            question={question}
            maxSteps={maxSteps}
            disabled={runState === "running" || !backendReady}
            onQuestionChange={setQuestion}
            onMaxStepsChange={setMaxSteps}
            onSubmit={handleStartQuery}
          />
        </section>
        <section className="panel">
          <h2>Run Output</h2>
          <RunStatus
            status={runState}
            result={result}
            error={errorMessage}
            events={events}
            showEvents={showEvents}
            onToggleEvents={handleToggleEvents}
          />
          {runState === "complete" && result && (
            <FeedbackPanel
              status={feedbackState}
              error={feedbackError}
              disabled={feedbackState === "sending"}
              onSubmit={handleFeedbackSubmit}
            />
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
