import { useParams } from 'react-router-dom'
import { formatDate, daysUntil } from '../../../lib/date'
import { useRuleDetail } from '../hooks/useRuleDetail'
import { RelatedRules } from '../components/RelatedRules'

export const RuleDetailPage = () => {
  const { ruleId } = useParams()
  const { rule, related, isLoading, error } = useRuleDetail(ruleId)

  if (isLoading) {
    return (
      <div className="card space-y-4 p-8">
        <div className="h-5 animate-pulse rounded bg-slate-200/70" />
        <div className="h-4 animate-pulse rounded bg-slate-200/70" />
        <div className="h-4 animate-pulse rounded bg-slate-200/70" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="card space-y-3 p-8 text-center">
        <h2 className="text-2xl font-semibold text-red-600">Unable to load regulation</h2>
        <p className="text-sm text-slate-600">{(error as Error).message}</p>
      </div>
    )
  }

  if (!rule) {
    return (
      <div className="card space-y-3 p-8 text-center">
        <h2 className="text-2xl font-semibold text-slate-900">Regulation not found</h2>
        <p className="text-sm text-slate-600">We could not locate the requested regulation. Try returning to search.</p>
      </div>
    )
  }

  const daysLeft = daysUntil(rule.comment_close_date)
  const isClosingSoon = typeof daysLeft === 'number' && daysLeft <= 21

  return (
    <div className="space-y-8">
      <article className="card space-y-6 p-6">
        <header className="space-y-2">
          <span className="rounded-full bg-brand-100 px-3 py-1 text-xs font-semibold text-brand-700">
            {rule.agency}
          </span>
          <h1 className="text-3xl font-semibold text-slate-900">{rule.title}</h1>
          <div className="flex flex-wrap items-center gap-3 text-sm text-slate-600">
            <span>Docket ID: {rule.docket_id}</span>
            <span className="hidden h-4 w-px bg-slate-200 sm:block" />
            <span>Stage: {rule.stage}</span>
          </div>
        </header>
        <section className="space-y-2">
          <h2 className="text-lg font-semibold text-slate-900">Summary</h2>
          <p className="text-sm text-slate-700 leading-relaxed">{rule.summary}</p>
        </section>
        <section className="space-y-2">
          <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-500">Comment window</h3>
          <div className="flex flex-wrap items-center gap-4 text-sm text-slate-600">
            <span>
              Opened: <strong className="text-slate-900">{formatDate(rule.comment_open_date)}</strong>
            </span>
            <span>
              Closes: <strong className="text-slate-900">{formatDate(rule.comment_close_date)}</strong>
            </span>
            {typeof daysLeft === 'number' ? (
              <span
                className={`rounded-full px-2.5 py-1 text-xs font-semibold ${
                  isClosingSoon ? 'bg-red-50 text-red-600' : 'bg-emerald-50 text-emerald-600'
                }`}
              >
                {daysLeft > 0 ? `${daysLeft} days remaining` : 'Comment period closed'}
              </span>
            ) : null}
          </div>
        </section>
        <section className="space-y-2">
          <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-500">Topics</h3>
          <div className="flex flex-wrap gap-2">
            {rule.topics.map((topic) => (
              <span key={topic} className="rounded-full border border-slate-200 bg-slate-100 px-3 py-1 text-xs font-medium text-slate-700">
                {topic}
              </span>
            ))}
          </div>
        </section>
        <section className="flex flex-wrap gap-3 text-sm">
          <a
            href={rule.comment_url}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-2 rounded-lg bg-brand-600 px-4 py-2 font-semibold text-white shadow-sm hover:bg-brand-500"
          >
            Submit a comment
          </a>
          <a
            href={rule.source_url}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-2 rounded-lg border border-slate-200 px-4 py-2 font-semibold text-slate-700 hover:border-brand-300 hover:text-brand-600"
          >
            View on Federal Register
          </a>
        </section>
      </article>

      <RelatedRules rules={related} />
    </div>
  )
}
