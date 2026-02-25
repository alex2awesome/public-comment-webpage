import { ReactNode, useEffect, useMemo, useRef } from "react";
import { RunState } from "../App";
import { FindingsSummary, RolloutResult, RolloutStreamEvent } from "../types";
import SummaryPanel from "./SummaryPanel";
import MemoWithCitations from "./MemoWithCitations";

export type MemoUpdateStatus = "idle" | "saving" | "success" | "error";

interface RunStatusProps {
  status: RunState;
  result: RolloutResult | null;
  error?: string | null;
  events: RolloutStreamEvent[];
  showEvents: boolean;
  maxSteps: number;
  onToggleEvents: () => void;
  planSummary: FindingsSummary | null;
  onSummaryUpdate: (summary: FindingsSummary, directives: string) => Promise<void>;
  memoUpdateStatus: MemoUpdateStatus;
  memoUpdateError?: string | null;
  memoHistory: string[];
  revisionIndex: number;
  revisionCount: number;
  onRevisionNavigate: (direction: "prev" | "next") => void;
}

const formatEventMeta = (event: RolloutStreamEvent) => {
  const meta: string[] = [];
  if (event.step !== undefined) {
    meta.push(`Step ${event.step}`);
  }
  if (event.tool) {
    meta.push(event.tool);
  }
  return meta.join(" • ");
};

const friendlyType = (type: string) => {
  const lookup: Record<string, string> = {
    run_started: "Run started",
    run_completed: "Run completed",
    complete: "Run completed",
    tool_call: "Tool call",
    tool_result: "Tool result",
    error: "Error",
  };
  return lookup[type] ?? type.replace(/[_-]+/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
};

const describeEvent = (event: RolloutStreamEvent) => {
  if (event.message) {
    return event.message;
  }
  if (event.tool) {
    return `Tool: ${event.tool}`;
  }
  if (event.question) {
    return event.question;
  }
  return "";
};

const renderMemoWithCitations = (memo: string, bib: RolloutResult["bib"]) => {
  const segments: ReactNode[] = [];
  let keyCounter = 0;
  const citeRegex = /<cite[^>]*id="([^"]+)"[^>]*>(.*?)<\/cite>/gi;

  const appendText = (text: string) => {
    if (!text) {
      return;
    }
    const parts = text.split(/\n/);
    parts.forEach((part, idx) => {
      segments.push(
        <span key={`memo-text-${keyCounter++}`}>
          {part}
          {idx < parts.length - 1 ? <br /> : null}
        </span>
      );
    });
  };

  let lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = citeRegex.exec(memo)) !== null) {
    if (match.index > lastIndex) {
      appendText(memo.slice(lastIndex, match.index));
    }
    const citeId = match[1];
    const inner = match[2];
    const bibEntry = bib.find((entry) => entry.paperId === citeId);
    const label = inner?.trim() || bibEntry?.title || citeId;
    if (bibEntry?.url) {
      segments.push(
        <a
          key={`memo-cite-${keyCounter++}`}
          href={bibEntry.url}
          target="_blank"
          rel="noreferrer"
          className="memo-cite"
          title={bibEntry.title ?? citeId}
        >
          {label}
        </a>
      );
    } else {
      segments.push(
        <span key={`memo-cite-${keyCounter++}`} className="memo-cite" title={bibEntry?.title ?? citeId}>
          {label}
        </span>
      );
    }
    lastIndex = citeRegex.lastIndex;
  }
  if (lastIndex < memo.length) {
    appendText(memo.slice(lastIndex));
  }
  return segments;
};

const RunStatus = ({
  status,
  result,
  error,
  events,
  showEvents,
  maxSteps,
  onToggleEvents,
  planSummary,
  onSummaryUpdate,
  memoUpdateStatus,
  memoUpdateError,
  memoHistory,
  revisionIndex,
  revisionCount,
  onRevisionNavigate,
}: RunStatusProps) => {
  const hasEvents = events.length > 0;
  const logBodyRef = useRef<HTMLDivElement | null>(null);
  const toolCallCount = useMemo(() => events.filter((event) => event.type === "tool_call").length, [events]);
  const completedSteps = Math.min(result?.steps ?? toolCallCount, maxSteps);
  const hasRevisions = revisionCount > 0;
  const displayRevisionCount = hasRevisions ? revisionCount : memoHistory.length || 1;
  const isLatestRevision = hasRevisions ? revisionIndex === revisionCount - 1 : true;

  useEffect(() => {
    if (!showEvents) {
      return;
    }
    const body = logBodyRef.current;
    if (body) {
      body.scrollTop = body.scrollHeight;
    }
  }, [events, showEvents]);

  const eventLog =
    hasEvents && showEvents ? (
      <div className="event-log">
        <div className="event-log-header">
          <h3>Live events</h3>
          <span className="event-progress">
            Step {completedSteps}/{maxSteps}
          </span>
        </div>
        <div className="event-log-body" ref={logBodyRef}>
          <ul>
            {events.map((event, idx) => (
              <li key={`${event.type}-${idx}`}>
                <div className="event-header">
                  <span className="event-type">{friendlyType(event.type)}</span>
                  {formatEventMeta(event) && <span className="event-meta">{formatEventMeta(event)}</span>}
                </div>
                {(() => {
                  const body = describeEvent(event);
                  return body ? <p className="event-body">{body}</p> : null;
                })()}
                {!event.message && event.result && (
                  (() => {
                    const resultStr = JSON.stringify(event.result);
                    return (
                      <p className="event-body result-json">
                        {resultStr.slice(0, 220)}
                        {resultStr.length > 220 ? "…" : ""}
                      </p>
                    );
                  })()
                )}
                {event.args && (
                  (() => {
                    const argsStr = JSON.stringify(event.args);
                    return (
                      <p className="event-body result-json">
                        Args: {argsStr.slice(0, 220)}
                        {argsStr.length > 220 ? "…" : ""}
                      </p>
                    );
                  })()
                )}
              </li>
            ))}
          </ul>
        </div>
      </div>
    ) : null;
  const eventToggle =
    hasEvents ? (
      <div className="event-toggle">
        <button type="button" className="secondary" onClick={onToggleEvents}>
          {showEvents ? "Hide steps" : "Show steps"}
        </button>
      </div>
    ) : null;

  if (status === "idle") {
    return (
      <>
        <div className="status-card note">Kick off a run to populate this panel.</div>
        {eventToggle}
        {eventLog}
      </>
    );
  }

  if (status === "running") {
    return (
      <>
        <div className="status-card note">Streaming tool calls… watch events below.</div>
        {eventToggle}
        {eventLog}
      </>
    );
  }

  if (status === "error") {
    return (
      <>
        <div className="status-card error">{error ?? "Something went wrong."}</div>
        {eventToggle}
        {eventLog}
      </>
    );
  }

  if (!result) {
    return (
      <>
        {eventToggle}
        {eventLog}
      </>
    );
  }

  return (
    <div>
      <div className="status-card success">
        <strong>Run complete.</strong> {result.steps} tool calls, reward {result.reward.toFixed(2)}.
      </div>
      <div className="result-meta">
        <span>Run ID: {result.runId}</span>
        <span>Task: {result.taskId}</span>
        <span>Tool calls: {result.toolCalls.length}</span>
      </div>
      <SummaryPanel
        summary={planSummary}
        disabled={status !== "complete"}
        status={memoUpdateStatus}
        error={memoUpdateError}
        onSubmit={onSummaryUpdate}
        revisionIndex={revisionIndex}
        revisionCount={displayRevisionCount}
        onNavigate={onRevisionNavigate}
        isLatestRevision={isLatestRevision}
      />
      <article className="memo-card">
        <div className="memo-card__header">
          <h3>Final memo</h3>
          {displayRevisionCount > 1 ? (
            <div className="memo-nav">
              <button type="button" className="secondary" onClick={() => onRevisionNavigate("prev")} disabled={revisionIndex === 0}>
                ←
              </button>
              <span>
                Version {revisionIndex + 1} / {displayRevisionCount}
              </span>
              <button
                type="button"
                className="secondary"
                onClick={() => onRevisionNavigate("next")}
                disabled={revisionIndex >= displayRevisionCount - 1}
              >
                →
              </button>
            </div>
          ) : null}
        </div>
        {(() => {
          const isViewingLatest = memoHistory.length === 0 || revisionIndex === memoHistory.length - 1;
          const hasStructuredMemo = Boolean(result.finalMemoBlocks && result.finalMemoBlocks.length > 0 && isViewingLatest);
          if (hasStructuredMemo) {
            return (
              <MemoWithCitations
                blocks={result.finalMemoBlocks}
                documents={result.sourceDocuments ?? []}
                fallbackMemo={result.finalMemo}
              />
            );
          }
          if (memoHistory.length > 0) {
            const memoText = memoHistory[Math.min(revisionIndex, memoHistory.length - 1)];
            return <div className="memo-body">{renderMemoWithCitations(memoText, result.bib)}</div>;
          }
          if (result.finalMemo) {
            return <div className="memo-body">{renderMemoWithCitations(result.finalMemo, result.bib)}</div>;
          }
          return <p>No memo submitted.</p>;
        })()}
      </article>
      <section>
        <h3>Bibliography</h3>
        {result.bib.length === 0 ? (
          <p className="status-card note">No bibliography entries captured this run.</p>
        ) : (
          <ul className="bib-list">
            {result.bib.map((entry, idx) => (
              <li key={entry.paperId || idx}>
                <strong>{entry.title || "Untitled"}</strong>
                {entry.year ? ` (${entry.year})` : null}
                <div>{entry.reason || "Reason not provided."}</div>
                {entry.url ? (
                  <a href={entry.url} target="_blank" rel="noreferrer">
                    {entry.url}
                  </a>
                ) : null}
              </li>
            ))}
          </ul>
        )}
      </section>
      {eventToggle}
      {eventLog}
    </div>
  );
};

export default RunStatus;
