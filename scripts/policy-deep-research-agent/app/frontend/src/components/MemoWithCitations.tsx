import { useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
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
      {citation?.cited_text ? <p className="citation-tooltip__quote">"{citation.cited_text.trim()}"</p> : null}
    </>
  );
};

/** Deduplicate citations by document_index, keeping the first occurrence. */
const uniqueCitationsByDoc = (citations: MemoCitation[]) => {
  const seen = new Set<number>();
  const result: MemoCitation[] = [];
  for (const c of citations) {
    const idx = c.document_index ?? -1;
    if (idx >= 0 && !seen.has(idx)) {
      seen.add(idx);
      result.push(c);
    }
  }
  return result;
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
      <div className="memo-body memo-body--markdown">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{fallbackMemo}</ReactMarkdown>
      </div>
    );
  }

  const renderCitationBadges = (blockIndex: number, citations: MemoCitation[]) => {
    const unique = uniqueCitationsByDoc(citations);
    if (unique.length === 0) {
      return null;
    }
    return (
      <span className="citation-badge-list">
        {unique.map((citation, idx) => {
          const docIndex = citation.document_index ?? -1;
          const doc = docMap.get(docIndex);
          const key = `${blockIndex}-${docIndex}-${idx}`;
          const label = (docIndex >= 0 ? labelMap.get(docIndex) : undefined) ?? "s?";
          return (
            <span
              key={key}
              className="citation-badge"
              data-active={activeKey === key}
              onMouseEnter={() => setActiveKey(key)}
              onMouseLeave={() => setActiveKey((current) => (current === key ? null : current))}
            >
              <span className="citation-badge__marker">{label}</span>
              <span
                className={`citation-tooltip${activeKey === key ? " visible" : ""}`}
                onMouseEnter={() => setActiveKey(key)}
              >
                {formatTooltipContent(doc ?? ({} as SourceDocument), citation)}
                {(doc?.kind === "argument" || doc?.kind === "recommendation") && citation?.cited_text ? (
                  <p className="citation-tooltip__quote">"{citation.cited_text.trim()}"</p>
                ) : null}
              </span>
            </span>
          );
        })}
      </span>
    );
  };

  // Concatenate all block text into one markdown string, inserting citation
  // badge placeholders at the end of each block's content so they appear
  // inline in the rendered output.
  //
  // We use a two-pass approach:
  //  1. Build the full markdown string with BADGE_PLACEHOLDER markers.
  //  2. Render via ReactMarkdown, replacing each marker with badge React nodes.

  const BADGE_PLACEHOLDER = "\u200B__CITE_BLOCK_";

  const fullMarkdown = useMemo(() => {
    // Blocks are sub-paragraph fragments — concatenate directly (no extra
    // line breaks) so the text's own whitespace controls paragraph structure.
    return workingBlocks
      .map((block, idx) => {
        const text = block.text ?? "";
        const hasCitations = block.citations && block.citations.length > 0;
        if (hasCitations) {
          return `${text}${BADGE_PLACEHOLDER}${idx}__`;
        }
        return text;
      })
      .join("");
  }, [workingBlocks]);

  // Regex to split rendered text nodes at badge placeholders.
  const badgePattern = useMemo(() => new RegExp(`${BADGE_PLACEHOLDER}(\\d+)__`), []);

  // Custom text renderer that injects citation badges at placeholder positions.
  // Text immediately before a badge placeholder is cited text and gets a
  // highlight background; all other text renders plain.
  const renderTextNode = (text: string) => {
    if (!badgePattern.test(text)) {
      return <>{text}</>;
    }
    // split produces: [text, capturedIdx, text, capturedIdx, …, text]
    const parts = text.split(new RegExp(`${BADGE_PLACEHOLDER}(\\d+)__`));
    const nodes: React.ReactNode[] = [];
    for (let i = 0; i < parts.length; i++) {
      if (i % 2 === 0) {
        // Text segment — highlighted if the very next part is a badge index
        const segment = parts[i];
        if (!segment) continue;
        const nextIsBadge = i + 1 < parts.length;
        if (nextIsBadge) {
          nodes.push(
            <span key={`h-${i}`} className="memo-citation-text">{segment}</span>
          );
        } else {
          nodes.push(<span key={`t-${i}`}>{segment}</span>);
        }
      } else {
        // Captured group: block index → render badges
        const blockIdx = parseInt(parts[i], 10);
        const block = workingBlocks[blockIdx];
        if (block) {
          nodes.push(
            <span key={`b-${blockIdx}`}>
              {renderCitationBadges(blockIdx, block.citations ?? [])}
            </span>
          );
        }
      }
    }
    return <>{nodes}</>;
  };

  return (
    <div className="memo-body memo-body--markdown">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          // Intercept every text node to replace badge placeholders
          p: ({ children, ...props }) => <p {...props}>{processChildren(children)}</p>,
          li: ({ children, ...props }) => <li {...props}>{processChildren(children)}</li>,
          td: ({ children, ...props }) => <td {...props}>{processChildren(children)}</td>,
          th: ({ children, ...props }) => <th {...props}>{processChildren(children)}</th>,
          h1: ({ children, ...props }) => <h1 {...props}>{processChildren(children)}</h1>,
          h2: ({ children, ...props }) => <h2 {...props}>{processChildren(children)}</h2>,
          h3: ({ children, ...props }) => <h3 {...props}>{processChildren(children)}</h3>,
          h4: ({ children, ...props }) => <h4 {...props}>{processChildren(children)}</h4>,
          blockquote: ({ children, ...props }) => <blockquote {...props}>{processChildren(children)}</blockquote>,
        }}
      >
        {fullMarkdown}
      </ReactMarkdown>
    </div>
  );

  function processChildren(children: React.ReactNode): React.ReactNode {
    if (typeof children === "string") {
      return renderTextNode(children);
    }
    if (Array.isArray(children)) {
      return children.map((child, i) => {
        if (typeof child === "string") {
          return <span key={i}>{renderTextNode(child)}</span>;
        }
        return child;
      });
    }
    return children;
  }
};

export default MemoWithCitations;
