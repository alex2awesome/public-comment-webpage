import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { FindingsSummary } from "../types";
import { API_BASE_URL } from "../lib/api";
import type { MemoUpdateStatus } from "./RunStatus";

type EditableArgument = { id: string; text: string; enabled: boolean };
type EditableArticle = {
  id: string;
  paperId?: string;
  title: string;
  authors: string[];
  url?: string;
  reason?: string;
  enabled: boolean;
};
type EditablePerson = { id: string; name: string; enabled: boolean; url?: string };

interface EditableSummary {
  arguments: EditableArgument[];
  articles: EditableArticle[];
  people: EditablePerson[];
}

type SectionKey = "arguments" | "articles" | "people" | "directives";

interface AuthorResult {
  authorId: string;
  name: string;
  url?: string;
  paperCount?: number;
  hIndex?: number;
}

interface SummaryPanelProps {
  summary: FindingsSummary | null;
  disabled: boolean;
  status: MemoUpdateStatus;
  error?: string | null;
  onSubmit: (summary: FindingsSummary, directives: string) => Promise<void>;
  revisionIndex: number;
  revisionCount: number;
  onNavigate: (direction: "prev" | "next") => void;
  isLatestRevision: boolean;
}

const randomId = () =>
  typeof crypto !== "undefined" && crypto.randomUUID ? crypto.randomUUID() : `id-${Date.now()}-${Math.random()}`;

const toEditable = (summary: FindingsSummary): EditableSummary => ({
  arguments: summary.topArguments.map((text, idx) => ({
    id: `${idx}-${text.slice(0, 8)}`,
    text,
    enabled: true,
  })),
  articles: summary.topArticles.map((article, idx) => ({
    id: article.paperId ?? `${idx}-${article.title?.slice(0, 8) ?? "article"}`,
    paperId: article.paperId,
    title: article.title ?? "Untitled article",
    authors: article.authors ?? [],
    url: article.url,
    reason: article.reason_chosen,
    enabled: true,
  })),
  people: summary.topPeople.map((name, idx) => ({
    id: `${idx}-${name.slice(0, 8)}`,
    name,
    enabled: true,
  })),
});

const fromEditable = (editable: EditableSummary): FindingsSummary => ({
  topArguments: editable.arguments.filter((arg) => arg.enabled && arg.text.trim()).map((arg) => arg.text.trim()),
  topArticles: editable.articles
    .filter((article) => article.enabled)
    .map((article) => ({
      id: article.id,
      paperId: article.paperId ?? article.id,
      title: article.title,
      authors: article.authors,
      url: article.url,
      reason_chosen: article.reason,
    })),
  topPeople: editable.people.filter((person) => person.enabled && person.name.trim()).map((person) => person.name.trim()),
});

const extractPaperId = (value: string): string | null => {
  const trimmed = value.trim();
  if (!trimmed) {
    return null;
  }
  try {
    const url = new URL(trimmed);
    if (url.hostname.includes("semanticscholar.org")) {
      const parts = url.pathname.split("/");
      return parts[parts.length - 1] || parts[parts.length - 2] || null;
    }
  } catch {
    // not a URL, fall through
  }
  return trimmed;
};

const SummaryPanel = ({ summary, disabled, status, error, onSubmit, revisionIndex, revisionCount, onNavigate, isLatestRevision }: SummaryPanelProps) => {
  const [expanded, setExpanded] = useState(true);
  const [editable, setEditable] = useState<EditableSummary | null>(null);
  const [directives, setDirectives] = useState("");
  const [newArgument, setNewArgument] = useState("");
  const [newPerson, setNewPerson] = useState("");
  const [paperInput, setPaperInput] = useState("");
  const [paperReason, setPaperReason] = useState("");
  const [articleStatus, setArticleStatus] = useState<"idle" | "loading" | "error">("idle");
  const [articleError, setArticleError] = useState<string | null>(null);
  const [sectionOpen, setSectionOpen] = useState<Record<SectionKey, boolean>>({
    arguments: false,
    articles: false,
    people: false,
    directives: false,
  });
  const [authorQuery, setAuthorQuery] = useState("");
  const [authorResults, setAuthorResults] = useState<AuthorResult[]>([]);
  const [authorStatus, setAuthorStatus] = useState<"idle" | "loading" | "error">("idle");
  const [authorError, setAuthorError] = useState<string | null>(null);
  const resolvedPeopleRef = useRef<Set<string>>(new Set());

  const editingLocked = disabled || !isLatestRevision;
  const displayRevisionCount = revisionCount || 1;

  useEffect(() => {
    if (summary) {
      setEditable(toEditable(summary));
      setExpanded(true);
    } else {
      setEditable(null);
    }
    setDirectives("");
    setSectionOpen({ arguments: false, articles: false, people: false, directives: false });
    setAuthorResults([]);
    setAuthorQuery("");
    setAuthorStatus("idle");
    setAuthorError(null);
    resolvedPeopleRef.current = new Set();
  }, [summary]);

  const selectedSummary = useMemo(() => {
    if (!editable) {
      return null;
    }
    return fromEditable(editable);
  }, [editable]);

  const canSubmit =
    !!selectedSummary &&
    (selectedSummary.topArguments.length > 0 || selectedSummary.topArticles.length > 0 || selectedSummary.topPeople.length > 0);

  const applyUpdates = async (event: FormEvent) => {
    event.preventDefault();
    if (!selectedSummary || editingLocked) {
      return;
    }
    await onSubmit(selectedSummary, directives.trim());
  };

  const addArgument = () => {
    if (!newArgument.trim() || !editable || editingLocked) {
      return;
    }
    setEditable({
      ...editable,
      arguments: [
        ...editable.arguments,
        {
          id: randomId(),
          text: newArgument.trim(),
          enabled: true,
        },
      ],
    });
    setNewArgument("");
  };

  const addPerson = () => {
    if (!newPerson.trim() || !editable || editingLocked) {
      return;
    }
    setEditable({
      ...editable,
      people: [
        ...editable.people,
        {
          id: randomId(),
          name: newPerson.trim(),
          enabled: true,
        },
      ],
    });
    setNewPerson("");
  };

  const addArticle = async () => {
    if (!editable || editingLocked) {
      return;
    }
    const paperId = extractPaperId(paperInput);
    if (!paperId) {
      setArticleError("Enter a Semantic Scholar paper ID or URL.");
      setArticleStatus("error");
      return;
    }
    setArticleStatus("loading");
    setArticleError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/semantic-scholar/paper/${encodeURIComponent(paperId)}`);
      if (!response.ok) {
        throw new Error("Semantic Scholar lookup failed.");
      }
      const data = await response.json();
      const authors = Array.isArray(data.authors) ? data.authors.map((author: any) => author.name).filter(Boolean) : [];
      const fallbackUrl = data.url ?? data.openAccessPdf?.url ?? (data.paperId ? `https://www.semanticscholar.org/paper/${data.paperId}` : undefined);
      const newArticle: EditableArticle = {
        id: data.paperId ?? paperId ?? randomId(),
        paperId: data.paperId ?? paperId,
        title: data.title ?? "Untitled article",
        authors,
        url: fallbackUrl,
        reason: paperReason.trim() || undefined,
        enabled: true,
      };
      setEditable({
        ...editable,
        articles: [...editable.articles, newArticle],
      });
      setPaperInput("");
      setPaperReason("");
      setArticleStatus("idle");
    } catch (err) {
      setArticleStatus("error");
      const message = err instanceof Error ? err.message : "Unable to fetch paper metadata.";
      setArticleError(message);
    }
  };

  const toggleSection = (key: SectionKey) => {
    setSectionOpen((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const searchAuthors = async () => {
    if (!authorQuery.trim() || editingLocked) {
      return;
    }
    setAuthorStatus("loading");
    setAuthorError(null);
    try {
      const response = await fetch(
        `${API_BASE_URL}/semantic-scholar/authors?query=${encodeURIComponent(authorQuery.trim())}&limit=5`
      );
      if (!response.ok) {
        throw new Error("Semantic Scholar author search failed.");
      }
      const data = await response.json();
      const results: AuthorResult[] = Array.isArray(data.data)
        ? data.data.map((author: any) => ({
            authorId: author.authorId ?? author.id ?? randomId(),
            name: author.name ?? "",
            url: author.url,
            paperCount: author.paperCount,
            hIndex: author.hIndex,
          }))
        : [];
      setAuthorResults(results.filter((author) => author.name));
      setAuthorStatus("idle");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unable to fetch authors.";
      setAuthorStatus("error");
      setAuthorError(message);
      setAuthorResults([]);
    }
  };

  const addAuthorFromSearch = (author: AuthorResult) => {
    if (!editable) {
      return;
    }
    const exists = editable.people.some((person) => person.id === author.authorId);
    if (exists) {
      return;
    }
    setEditable({
      ...editable,
      people: [
        ...editable.people,
        {
          id: author.authorId || randomId(),
          name: author.name,
          enabled: true,
          url: author.url,
        },
      ],
    });
  };

  let statusMessage: string | null = null;
  if (status === "saving") {
    statusMessage = "Updating memo…";
  } else if (status === "success") {
    statusMessage = "Memo updated with your changes.";
  } else if (status === "error") {
    statusMessage = error ?? "Unable to update memo.";
  }
  if (!isLatestRevision && !disabled) {
    statusMessage = `Viewing revision ${revisionIndex + 1} of ${displayRevisionCount}. Navigate to the latest to make edits.`;
  }

  const fetchAuthorProfile = async (name: string): Promise<AuthorResult | null> => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/semantic-scholar/authors?query=${encodeURIComponent(name)}&limit=1`
      );
      if (!response.ok) {
        return null;
      }
      const data = await response.json();
      const entry = Array.isArray(data.data) && data.data.length > 0 ? data.data[0] : null;
      if (!entry || !entry.name) {
        return null;
      }
      return {
        authorId: entry.authorId ?? entry.id ?? name,
        name: entry.name,
        url: entry.url,
        paperCount: entry.paperCount,
        hIndex: entry.hIndex,
      };
    } catch {
      return null;
    }
  };

  useEffect(() => {
    if (!editable) {
      return;
    }
    editable.people.forEach((person) => {
      if (person.url || resolvedPeopleRef.current.has(person.id)) {
        return;
      }
      resolvedPeopleRef.current.add(person.id);
      void (async () => {
        const profile = await fetchAuthorProfile(person.name);
        if (profile) {
          setEditable((prev) => {
            if (!prev) {
              return prev;
            }
            return {
              ...prev,
              people: prev.people.map((entry) =>
                entry.id === person.id
                  ? { ...entry, name: profile.name ?? entry.name, url: profile.url ?? entry.url }
                  : entry
              ),
            };
          });
        }
      })();
    });
  }, [editable]);

  return (
    <section className="summary-panel">
      <div className="summary-panel__header">
        <div>
          <p className="eyebrow">Memo Plan & Overview</p>
          <h3>Memo Plan & Overview</h3>
          <p className="section-copy">
            Review the planned arguments, sources, and contributors before regenerating the memo.
          </p>
        </div>
        <div className="summary-panel__header-controls">
          {revisionCount > 1 ? (
            <div className="memo-nav">
              <button type="button" className="secondary" onClick={() => onNavigate("prev")} disabled={revisionIndex === 0}>
                ←
              </button>
              <span>
                Version {revisionIndex + 1} / {displayRevisionCount}
              </span>
              <button
                type="button"
                className="secondary"
                onClick={() => onNavigate("next")}
                disabled={revisionIndex >= revisionCount - 1}
              >
                →
              </button>
            </div>
          ) : null}
          <button type="button" className="secondary" onClick={() => setExpanded((prev) => !prev)}>
            {expanded ? "Hide details" : "Show details"}
          </button>
        </div>
      </div>
      {!summary ? (
        <p className="status-card note">Waiting for the agent to call summarize_findings… hang tight!</p>
      ) : null}
      {summary && expanded && editable ? (
        <>
          {!isLatestRevision && !disabled ? (
            <p className="status-card note summary-note">
              Viewing revision {revisionIndex + 1} of {displayRevisionCount}. Switch to the latest version to edit the plan.
            </p>
          ) : null}
        <form className="summary-panel__body" onSubmit={applyUpdates}>
          <div className="summary-section">
            <button type="button" className="summary-section__toggle" onClick={() => toggleSection("arguments")}>
              <div>
                <h4>Top arguments</h4>
                <p className="section-copy">Define the 3–5 core claims you will rely on.</p>
              </div>
              <span className="summary-section__chevron">{sectionOpen.arguments ? "−" : "+"}</span>
            </button>
            {sectionOpen.arguments && (
              <div className="summary-section__body">
                <ul className="summary-card-list">
                  {editable.arguments.map((argument) => (
                    <li key={argument.id} className="summary-card">
                      <label className="summary-card__row">
                        <input
                          type="checkbox"
                          checked={argument.enabled}
                          disabled={editingLocked}
                          onChange={(event) => {
                            const next = editable.arguments.map((item) =>
                              item.id === argument.id ? { ...item, enabled: event.target.checked } : item
                            );
                            setEditable({ ...editable, arguments: next });
                          }}
                        />
                        <textarea
                          disabled={editingLocked}
                          value={argument.text}
                          onChange={(event) => {
                            const next = editable.arguments.map((item) =>
                              item.id === argument.id ? { ...item, text: event.target.value } : item
                            );
                            setEditable({ ...editable, arguments: next });
                          }}
                        />
                      </label>
                    </li>
                  ))}
                </ul>
                <div className="summary-panel__add-row">
                  <input
                    type="text"
                    placeholder="Add a new argument"
                    value={newArgument}
                    disabled={editingLocked}
                    onChange={(event) => setNewArgument(event.target.value)}
                  />
                  <button type="button" className="secondary" disabled={editingLocked || !newArgument.trim()} onClick={addArgument}>
                    Add argument
                  </button>
                </div>
              </div>
            )}
          </div>
          <div className="summary-section">
            <button type="button" className="summary-section__toggle" onClick={() => toggleSection("articles")}>
              <div>
                <h4>Top articles</h4>
                <p className="section-copy">List the references that justify those arguments.</p>
              </div>
              <span className="summary-section__chevron">{sectionOpen.articles ? "−" : "+"}</span>
            </button>
            {sectionOpen.articles && (
              <div className="summary-section__body">
                <ul className="summary-card-list article-list">
                  {editable.articles.map((article) => (
                    <li key={article.id} className="summary-card article-card">
                      <label className="article-card__header">
                        <input
                          type="checkbox"
                          checked={article.enabled}
                          disabled={editingLocked}
                          onChange={(event) => {
                            const next = editable.articles.map((item) =>
                              item.id === article.id ? { ...item, enabled: event.target.checked } : item
                            );
                            setEditable({ ...editable, articles: next });
                          }}
                        />
                        <div>
                          <strong>{article.title}</strong>
                          {article.url ? (
                            <a href={article.url} target="_blank" rel="noreferrer">
                              {article.url}
                            </a>
                          ) : null}
                          {article.authors.length > 0 ? (
                            <p className="article-card__authors">{article.authors.join(", ")}</p>
                          ) : null}
                        </div>
                      </label>
                      <textarea
                        placeholder="Reason for including this source"
                        disabled={editingLocked}
                        value={article.reason ?? ""}
                        onChange={(event) => {
                          const next = editable.articles.map((item) =>
                            item.id === article.id ? { ...item, reason: event.target.value } : item
                          );
                          setEditable({ ...editable, articles: next });
                        }}
                      />
                    </li>
                  ))}
                </ul>
                <div className="summary-panel__add-article">
                  <input
                    type="text"
                    placeholder="Semantic Scholar paper URL or ID"
                    value={paperInput}
                    disabled={editingLocked || articleStatus === "loading"}
                    onChange={(event) => setPaperInput(event.target.value)}
                  />
                  <textarea
                    placeholder="Brief reason for adding this paper"
                    value={paperReason}
                    disabled={editingLocked || articleStatus === "loading"}
                    onChange={(event) => setPaperReason(event.target.value)}
                  />
                  <button
                    type="button"
                    className="secondary"
                    disabled={editingLocked || articleStatus === "loading" || !paperInput.trim()}
                    onClick={addArticle}
                  >
                    {articleStatus === "loading" ? "Adding…" : "Add paper"}
                  </button>
                  {articleStatus === "error" && articleError ? <p className="form-error">{articleError}</p> : null}
                </div>
              </div>
            )}
          </div>
          <div className="summary-section">
            <button type="button" className="summary-section__toggle" onClick={() => toggleSection("people")}>
              <div>
                <h4>Top contributors</h4>
                <p className="section-copy">Search Semantic Scholar for authors or experts to feature.</p>
              </div>
              <span className="summary-section__chevron">{sectionOpen.people ? "−" : "+"}</span>
            </button>
            {sectionOpen.people && (
              <div className="summary-section__body">
                <ul className="summary-card-list">
                  {editable.people.map((person) => (
                    <li key={person.id} className="summary-card">
                      <label className="summary-card__row">
                        <input
                          type="checkbox"
                          checked={person.enabled}
                          disabled={editingLocked}
                          onChange={(event) => {
                            const next = editable.people.map((item) =>
                              item.id === person.id ? { ...item, enabled: event.target.checked } : item
                            );
                            setEditable({ ...editable, people: next });
                          }}
                        />
                      <input
                        type="text"
                        value={person.name}
                        disabled={editingLocked}
                          onChange={(event) => {
                            const next = editable.people.map((item) =>
                              item.id === person.id ? { ...item, name: event.target.value } : item
                            );
                            setEditable({ ...editable, people: next });
                          }}
                        />
                      </label>
                      {person.url ? (
                        <a href={person.url} target="_blank" rel="noreferrer" className="summary-card__link">
                          {person.url}
                        </a>
                      ) : null}
                    </li>
                  ))}
                </ul>
                <div className="summary-panel__add-row">
                <input
                  type="text"
                  placeholder="Add a contributor manually"
                  value={newPerson}
                  disabled={editingLocked}
                  onChange={(event) => setNewPerson(event.target.value)}
                />
                <button type="button" className="secondary" disabled={editingLocked || !newPerson.trim()} onClick={addPerson}>
                    Add person
                  </button>
                </div>
                <div className="summary-panel__author-search">
                <input
                  type="text"
                  placeholder="Search Semantic Scholar authors"
                  value={authorQuery}
                  disabled={editingLocked || authorStatus === "loading"}
                  onChange={(event) => setAuthorQuery(event.target.value)}
                />
                <button
                  type="button"
                  className="secondary"
                  disabled={editingLocked || !authorQuery.trim() || authorStatus === "loading"}
                  onClick={searchAuthors}
                >
                    {authorStatus === "loading" ? "Searching…" : "Search"}
                  </button>
                </div>
                {authorStatus === "error" && authorError ? <p className="form-error">{authorError}</p> : null}
                {authorResults.length > 0 ? (
                  <ul className="author-results">
                    {authorResults.map((author) => (
                      <li key={author.authorId} className="summary-card author-card">
                        <div>
                          <strong>{author.name}</strong>
                          {author.url ? (
                            <a href={author.url} target="_blank" rel="noreferrer">
                              {author.url}
                            </a>
                          ) : null}
                          <p className="author-card__meta">
                            {author.paperCount ? `${author.paperCount} papers` : null}
                            {author.paperCount && author.hIndex ? " • " : null}
                            {author.hIndex ? `h-index ${author.hIndex}` : null}
                          </p>
                        </div>
                          <button
                            type="button"
                            className="secondary"
                            disabled={editingLocked}
                            onClick={() => addAuthorFromSearch(author)}
                          >
                          Add contributor
                        </button>
                      </li>
                    ))}
                  </ul>
                ) : null}
              </div>
            )}
          </div>
          <div className="summary-section">
            <button type="button" className="summary-section__toggle" onClick={() => toggleSection("directives")}>
              <div>
                <h4>Additional directives</h4>
                <p className="section-copy">Provide extra guidance on tone, structure, or stakeholder focus.</p>
              </div>
              <span className="summary-section__chevron">{sectionOpen.directives ? "−" : "+"}</span>
            </button>
            {sectionOpen.directives && (
              <div className="summary-section__body">
                <textarea
                  className="summary-directives"
                  placeholder="e.g., Emphasize FEMA funding pathways and keep the intro to two paragraphs."
                  value={directives}
                  disabled={editingLocked}
                  onChange={(event) => setDirectives(event.target.value)}
                />
              </div>
            )}
          </div>
          <div className="summary-panel__actions">
            <button type="submit" className="primary" disabled={editingLocked || status === "saving" || !canSubmit}>
              {status === "saving" ? "Updating…" : "Update report"}
            </button>
            {statusMessage ? <span className={`summary-panel__status summary-panel__status--${status}`}>{statusMessage}</span> : null}
          </div>
          {disabled ? (
            <p className="summary-panel__hint">Editing will unlock once the run completes and the memo is ready.</p>
          ) : null}
        </form>
        </>
      ) : null}
    </section>
  );
};

export default SummaryPanel;
