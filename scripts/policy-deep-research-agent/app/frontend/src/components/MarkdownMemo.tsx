import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import type { RolloutResult } from "../types";

interface MarkdownMemoProps {
  memo: string;
  bib: RolloutResult["bib"];
}

/**
 * Renders a memo string as Markdown.
 * <cite id="PAPER_ID">Text</cite> tags pass through via rehype-raw
 * and are rendered as linked references into the bibliography.
 */
const MarkdownMemo = ({ memo, bib }: MarkdownMemoProps) => {
  return (
    <div className="memo-body memo-body--markdown">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeRaw]}
        components={{
          cite: ({ node, ...props }) => {
            const id = (props as Record<string, unknown>).id as string | undefined;
            const bibEntry = id ? bib.find((entry) => entry.paperId === id) : undefined;
            const children = (props as Record<string, unknown>).children;
            if (bibEntry?.url) {
              return (
                <a
                  href={bibEntry.url}
                  target="_blank"
                  rel="noreferrer"
                  className="memo-cite"
                  title={bibEntry.title ?? id}
                >
                  {children as React.ReactNode}
                </a>
              );
            }
            return (
              <span className="memo-cite" title={bibEntry?.title ?? id}>
                {children as React.ReactNode}
              </span>
            );
          },
        }}
      />
    </div>
  );
};

export default MarkdownMemo;
