import { RuleCard } from '../../../components/RuleCard'
import { Pagination } from '../../../components/Pagination'
import type { SearchResponse } from '../../../lib/types'

interface SearchResultsProps {
  data?: SearchResponse
  isLoading: boolean
  isFetching: boolean
  error?: Error | null
  onPageChange: (page: number) => void
}

export const SearchResults = ({
  data,
  isLoading,
  isFetching,
  error,
  onPageChange,
}: SearchResultsProps) => {
  if (isLoading) {
    return (
      <div className="card flex flex-col gap-4 p-8">
        <SkeletonRow label="Loading search results" />
        <SkeletonRow />
        <SkeletonRow />
      </div>
    )
  }

  if (error) {
    return (
      <div className="card space-y-3 p-8 text-center">
        <h3 className="text-lg font-semibold text-red-600">Something went wrong</h3>
        <p className="text-sm text-slate-600">{error.message}</p>
      </div>
    )
  }

  if (!data || data.items.length === 0) {
    return (
      <div className="card space-y-3 p-10 text-center">
        <h3 className="text-lg font-semibold text-slate-800">No regulations match your filters yet</h3>
        <p className="text-sm text-slate-600">
          Try broadening your search terms or adjusting the agency, topic, or comment window filters.
        </p>
      </div>
    )
  }

  return (
    <div className="card flex flex-col gap-6 p-6">
      <div className="flex items-center justify-between text-sm text-slate-600">
        <span>
          Showing <span className="font-semibold text-slate-900">{data.items.length}</span> of{' '}
          <span className="font-semibold text-slate-900">{data.total}</span> results
        </span>
        {isFetching ? <span className="text-xs text-slate-400">Refreshing…</span> : null}
      </div>
      <div className="flex flex-col gap-4">
        {data.items.map((rule) => (
          <RuleCard key={rule.id} rule={rule} />
        ))}
      </div>
      <Pagination page={data.page} totalPages={data.totalPages} onChange={onPageChange} isLoading={isFetching} />
    </div>
  )
}

const SkeletonRow = ({ label }: { label?: string }) => (
  <div className="space-y-2">
    <div className="h-4 rounded bg-slate-200/70" />
    <div className="h-3 rounded bg-slate-200/70" />
    {label ? <div className="text-xs text-slate-400">{label}</div> : null}
  </div>
)
