import { ReactNode } from "react";
import { RunState } from "../App";
import { RolloutResult, RolloutStreamEvent } from "../types";

interface RunStatusProps {
  status: RunState;
  result: RolloutResult | null;
  error?: string | null;
  events: RolloutStreamEvent[];
  showEvents: boolean;
  onToggleEvents: () => void;
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

const RunStatus = ({ status, result, error, events, showEvents, onToggleEvents }: RunStatusProps) => {
  const hasEvents = events.length > 0;
  const eventLog =
    hasEvents && showEvents ? (
      <div className="event-log">
        <h3>Live events</h3>
        <ul>
          {events.map((event, idx) => (
            <li key={`${event.type}-${idx}`}>
              <div className="event-header">
                <span className="event-type">{event.type}</span>
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
      <article className="memo-card">
        <h3>Final memo</h3>
        {result.finalMemo ? (
          <div className="memo-body">{renderMemoWithCitations(result.finalMemo, result.bib)}</div>
        ) : (
          <p>No memo submitted.</p>
        )}
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
