import { ReactNode, useMemo, useState } from "react";
import { normalizeMemoBlockCitations } from "../lib/citations";
import { MemoBlock, MemoCitation, SourceDocument } from "../types";

interface MemoWithCitationsProps {
  blocks?: MemoBlock[] | null;
  documents: SourceDocument[];
  fallbackMemo: string;
}

const formatTooltipContent = (doc: SourceDocument, citation?: MemoCitation) => {
  if (!doc) {
    return null;
  }
  if (doc.kind === "person") {
    return (
      <>
        <div className="citation-tooltip__title">
          <span role="img" aria-label="Person">
            👤
          </span>
          <strong>{doc.title ?? doc.document_id ?? "Referenced person"}</strong>
        </div>
        <p>{doc.title ?? doc.reason ?? "Stakeholder highlighted in the summary."}</p>
      </>
    );
  }
  if (doc.kind === "argument") {
    return (
      <>
        <div className="citation-tooltip__title argument">
          Argument #{doc.argument_index ?? "?"}
        </div>
        <p>{doc.reason ?? doc.title ?? "Key argument captured in summarize_findings."}</p>
      </>
    );
  }
  if (doc.kind === "recommendation") {
    return (
      <>
        <div className="citation-tooltip__title recommendation">
          Recommendation #{doc.recommendation_index ?? "?"}
        </div>
        <p>{doc.reason ?? doc.title ?? "Priority recommendation captured in summarize_findings."}</p>
      </>
    );
  }
  const title = doc.title ?? "Source document";
  return (
    <>
      <div className="citation-tooltip__title">
        <strong>{title}</strong>
      </div>
      {doc.authors ? <p className="citation-tooltip__meta">{doc.authors}</p> : null}
      {doc.reason ? <p>{doc.reason}</p> : null}
      {doc.url ? (
        <a href={doc.url} target="_blank" rel="noreferrer">
          View source
        </a>
      ) : null}
      {citation?.cited_text ? <p className="citation-tooltip__quote">“{citation.cited_text.trim()}”</p> : null}
    </>
  );
};

const MemoWithCitations = ({ blocks, documents, fallbackMemo }: MemoWithCitationsProps) => {
  const docMap = useMemo(() => {
    const map = new Map<number, SourceDocument>();
    documents.forEach((doc) => {
      map.set(doc.document_index, doc);
    });
    return map;
  }, [documents]);

  const [activeKey, setActiveKey] = useState<string | null>(null);

  const labelMap = useMemo(() => {
    const map = new Map<number, string>();
    let sourceCounter = 1;
    let argumentCounter = 1;
    let personCounter = 1;
    let recommendationCounter = 1;

    documents.forEach((doc) => {
      if (typeof doc.document_index !== "number") {
        return;
      }
      if (doc.kind === "argument") {
        if (typeof doc.argument_index === "number" && !Number.isNaN(doc.argument_index)) {
          map.set(doc.document_index, `a${doc.argument_index}`);
          argumentCounter = Math.max(argumentCounter, doc.argument_index + 1);
        } else {
          map.set(doc.document_index, `a${argumentCounter}`);
          argumentCounter += 1;
        }
        return;
      }
      if (doc.kind === "recommendation") {
        if (typeof doc.recommendation_index === "number" && !Number.isNaN(doc.recommendation_index)) {
          map.set(doc.document_index, `r${doc.recommendation_index}`);
          recommendationCounter = Math.max(recommendationCounter, doc.recommendation_index + 1);
        } else {
          map.set(doc.document_index, `r${recommendationCounter}`);
          recommendationCounter += 1;
        }
        return;
      }
      if (doc.kind === "person") {
        if (typeof doc.person_index === "number" && !Number.isNaN(doc.person_index)) {
          map.set(doc.document_index, `p${doc.person_index}`);
          personCounter = Math.max(personCounter, doc.person_index + 1);
        } else {
          map.set(doc.document_index, `p${personCounter}`);
          personCounter += 1;
        }
        return;
      }
      map.set(doc.document_index, `s${sourceCounter++}`);
    });

    return map;
  }, [documents]);

  const normalizedBlocks = useMemo(() => {
    if (!blocks) {
      return null;
    }
    return normalizeMemoBlockCitations(blocks);
  }, [blocks]);

  const workingBlocks = normalizedBlocks ?? blocks;

  if (!workingBlocks || workingBlocks.length === 0) {
    return (
      <div className="memo-body memo-body--structured">
        <div className="memo-paragraph">{fallbackMemo}</div>
      </div>
    );
  }

  const renderTextWithCitations = (text: string, blockIndex: number, citations: MemoCitation[]) => {
    if (!citations || citations.length === 0) {
      return text;
    }
    const parts: ReactNode[] = [];
    let lastIndex = 0;
    const sortedCitations = [...citations].sort((a, b) => (a.start_char_index ?? 0) - (b.start_char_index ?? 0));

    sortedCitations.forEach((citation, idx) => {
      const start = citation.start_char_index ?? 0;
      const end = citation.end_char_index ?? start;
      if (start > lastIndex) {
        parts.push(text.slice(lastIndex, start));
      }
      const doc = docMap.get(citation.document_index ?? -1);
      const key = `${blockIndex}-${citation.document_index}-${idx}`;
      const label =
        (doc?.document_index !== undefined ? labelMap.get(doc.document_index) : undefined) ?? `s?`;
      parts.push(
        <span
          key={key}
          className="memo-citation-span"
          onMouseLeave={() => setActiveKey((current) => (current === key ? null : current))}
        >
          <span className="memo-citation-text">{text.slice(start, end)}</span>
          <span className="citation-inline">
            <span
              className="citation-badge"
              data-active={activeKey === key}
              onMouseEnter={() => setActiveKey(key)}
            >
              <span className="citation-badge__marker">{label}</span>
              <span
                className={`citation-tooltip${activeKey === key ? " visible" : ""}`}
                onMouseEnter={() => setActiveKey(key)}
              >
                {formatTooltipContent(doc ?? ({} as SourceDocument), citation)}
                {(doc?.kind === "argument" || doc?.kind === "recommendation") && citation?.cited_text ? (
                  <p className="citation-tooltip__quote">“{citation.cited_text.trim()}”</p>
                ) : null}
              </span>
            </span>
          </span>
        </span>
      );
      lastIndex = end;
    });
    if (lastIndex < text.length) {
      parts.push(text.slice(lastIndex));
    }
    return parts;
  };

  return (
    <div className="memo-body memo-body--structured">
      {workingBlocks.map((block, idx) => {
        const text = block.text ?? "";
        const hasCitations = Boolean(block.citations && block.citations.length > 0);
        const paragraphClass = hasCitations ? "memo-paragraph has-citation" : "memo-paragraph";
        return (
          <div key={idx} className={paragraphClass}>
            {renderTextWithCitations(text, idx, block.citations ?? [])}
          </div>
        );
      })}
    </div>
  );
};

export default MemoWithCitations;
