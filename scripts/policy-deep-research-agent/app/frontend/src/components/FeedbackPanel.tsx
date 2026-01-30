import { FormEvent, useState } from "react";
import { FeedbackSentiment } from "../types";

export interface FeedbackFormValues {
  sentiment: FeedbackSentiment;
  note: string;
}

interface FeedbackPanelProps {
  status: "idle" | "sending" | "sent" | "error";
  error?: string | null;
  disabled?: boolean;
  onSubmit: (values: FeedbackFormValues) => Promise<void> | void;
}

const FeedbackPanel = ({ status, error, disabled, onSubmit }: FeedbackPanelProps) => {
  const [sentiment, setSentiment] = useState<FeedbackSentiment | null>(null);
  const [note, setNote] = useState("");
  const [localError, setLocalError] = useState<string | null>(null);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (!sentiment) {
      setLocalError("Select thumbs up or thumbs down before sending feedback.");
      return;
    }
    setLocalError(null);
    await onSubmit({ sentiment, note: note.trim() });
  };

  const isSending = status === "sending";
  const isDone = status === "sent";

  return (
    <div className="feedback-panel">
      <h3>Send LangSmith Feedback</h3>
      <p className="helper">
        Label the run directly from the UI. The backend will forward this payload to LangSmith so traces show real
        user sentiment.
      </p>
      <form onSubmit={handleSubmit}>
        <div className="feedback-row">
          <button
            type="button"
            className={sentiment === "positive" ? "active" : ""}
            onClick={() => setSentiment("positive")}
            disabled={isSending || disabled}
          >
            👍 Helpful
          </button>
          <button
            type="button"
            className={sentiment === "negative" ? "active" : ""}
            onClick={() => setSentiment("negative")}
            disabled={isSending || disabled}
          >
            👎 Needs work
          </button>
        </div>
        <textarea
          placeholder="Optional context for the reviewer…"
          value={note}
          disabled={isSending || disabled}
          onChange={(event) => setNote(event.target.value)}
        />
        {localError && <div className="status-card error">{localError}</div>}
        {error && status === "error" && <div className="status-card error">{error}</div>}
        {isDone && <div className="status-card success">Feedback saved to LangSmith. Thank you!</div>}
        <button className="primary" type="submit" disabled={isSending || disabled || isDone}>
          {isSending ? "Sending…" : isDone ? "Sent" : "Submit feedback"}
        </button>
      </form>
    </div>
  );
};

export default FeedbackPanel;
