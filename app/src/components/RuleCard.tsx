import { useState } from 'react'
import { Link } from 'react-router-dom'
import { daysUntil, formatDate } from '../lib/date'
import { isCommentPeriodActive } from '../lib/rules'
import type { Rule } from '../lib/types'

interface RuleCardProps {
  rule: Rule
  showActions?: boolean
}

const formatBoolean = (value: boolean) => (value ? 'Active' : 'Inactive')

const displayValue = (value?: string | null) => (value && value.trim().length > 0 ? value : '—')

export const RuleCard = ({ rule, showActions = true }: RuleCardProps) => {
  const [showSupplementary, setShowSupplementary] = useState(false)
  const daysLeft = daysUntil(rule.comment_due_date ?? rule.comment_close_date)
  const isClosingSoon = typeof daysLeft === 'number' && daysLeft <= 21
  const isActive = isCommentPeriodActive(rule)

  return (
    <article className="card flex flex-col gap-6 p-6">
      <header className="flex flex-col gap-3">
        <div className="flex flex-wrap items-center gap-3 text-xs font-semibold uppercase tracking-wide">
          <span className="rounded-full bg-brand-100 px-3 py-1 text-brand-700">{displayValue(rule.type)}</span>
          {rule.is_rfi_rfc && rule.rfi_rfc_label ? (
            <span className="rounded-full bg-indigo-100 px-3 py-1 text-indigo-700">{rule.rfi_rfc_label}</span>
          ) : null}
          {rule.comment_status ? (
            <span className="rounded-full bg-slate-100 px-3 py-1 text-slate-600">{rule.comment_status}</span>
          ) : null}
          <span
            className={`rounded-full px-3 py-1 text-xs font-semibold ${
              isActive ? 'bg-emerald-50 text-emerald-600' : 'bg-slate-200 text-slate-700'
            }`}
          >
            {formatBoolean(isActive)}
          </span>
        </div>
        <div className="flex flex-col gap-1">
          <Link to={`/rules/${rule.id}`} className="text-xl font-semibold text-slate-900 hover:text-brand-600">
            {rule.title}
          </Link>
          <p className="text-sm text-slate-600">{displayValue(rule.agency)}</p>
        </div>
      </header>

      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        <MetadataItem label="Publication" value={formatDate(rule.publication_date ?? rule.comment_open_date)} />
        <MetadataItem label="Comment opens" value={formatDate(rule.comment_start_date ?? rule.comment_open_date)} />
        <MetadataItem label="Comment due" value={formatDate(rule.comment_due_date ?? rule.comment_close_date)} />
      </section>

      <section className="space-y-3 text-sm text-slate-700">
        {rule.summary ? <p>{rule.summary}</p> : null}
        {rule.details && rule.details !== rule.summary ? (
          <p className="text-slate-600">{rule.details}</p>
        ) : null}
        {rule.abstract && rule.abstract !== rule.summary ? (
          <p className="text-slate-600">{rule.abstract}</p>
        ) : null}
      </section>

      {rule.supplementary_information ? (
        <details
          className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700"
          open={showSupplementary}
          onToggle={(event) => setShowSupplementary(event.currentTarget.open)}
        >
          <summary className="cursor-pointer font-semibold text-slate-800">Supplementary information</summary>
          <p className="mt-2 whitespace-pre-line text-slate-600">{rule.supplementary_information}</p>
        </details>
      ) : null}

      <footer className="flex flex-wrap items-center gap-3 text-sm text-slate-600">
        {typeof daysLeft === 'number' ? (
          <span
            className={`rounded-full px-2.5 py-1 text-xs font-semibold ${
              daysLeft > 0
                ? isClosingSoon
                  ? 'bg-red-50 text-red-600'
                  : 'bg-amber-50 text-amber-700'
                : 'bg-slate-200 text-slate-600'
            }`}
          >
            {daysLeft > 0 ? `${daysLeft} days remaining` : 'Comment period closed'}
          </span>
        ) : null}

        {showActions ? (
          <div className="flex flex-wrap items-center gap-3">
            <a
              href={rule.comment_url}
              target="_blank"
              rel="noreferrer"
              className="font-semibold text-brand-600 hover:text-brand-500"
            >
              Submit comment
            </a>
            <a
              href={rule.source_url}
              target="_blank"
              rel="noreferrer"
              className="font-semibold text-slate-600 hover:text-brand-500"
            >
              Federal Register notice
            </a>
          </div>
        ) : null}

        <div className="flex flex-wrap items-center gap-4 text-xs text-slate-500">
          {rule.docket_id && rule.docket_id !== 'N/A' ? <span>Docket: {rule.docket_id}</span> : null}
          {rule.fr_document_number ? <span>FR Doc: {rule.fr_document_number}</span> : null}
        </div>
      </footer>
    </article>
  )
}

const MetadataItem = ({ label, value }: { label: string; value: string }) => (
  <div className="flex flex-col gap-1">
    <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">{label}</span>
    <span className="text-sm font-medium text-slate-800">{value}</span>
  </div>
)
