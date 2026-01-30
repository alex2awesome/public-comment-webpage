import { FormEvent } from "react";

interface QueryFormProps {
  question: string;
  maxSteps: number;
  disabled?: boolean;
  onQuestionChange: (value: string) => void;
  onMaxStepsChange: (value: number) => void;
  onSubmit: () => void;
}

const QueryForm = ({
  question,
  maxSteps,
  disabled,
  onQuestionChange,
  onMaxStepsChange,
  onSubmit,
}: QueryFormProps) => {
  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    if (!disabled) {
      onSubmit();
    }
  };

  return (
    <form className="query-form" onSubmit={handleSubmit}>
      <div className="form-group">
        <label htmlFor="question">Policy question</label>
        <textarea
          id="question"
          placeholder="How are US states using carbon pricing to fund climate resilience?"
          value={question}
          disabled={disabled}
          onChange={(event) => onQuestionChange(event.target.value)}
        />
      </div>

      <div className="form-group">
        <label htmlFor="maxSteps">Max tool calls</label>
        <input
          id="maxSteps"
          type="number"
          min={4}
          max={24}
          step={1}
          value={maxSteps}
          disabled={disabled}
          onChange={(event) => {
            const next = Number(event.target.value);
            onMaxStepsChange(Number.isNaN(next) ? 4 : next);
          }}
        />
        <span className="helper">The agent will auto-submit on the last step.</span>
      </div>

      <button className="primary" type="submit" disabled={disabled}>
        {disabled ? "Running…" : "Start Query"}
      </button>
    </form>
  );
};

export default QueryForm;
