import { useEffect, useMemo, useState } from 'react'
import { SearchResults } from '../components/SearchResults'
import { useSearchFilters } from '../hooks/useSearchFilters'
import { useSearchRules } from '../hooks/useSearchRules'
import { useDebouncedValue } from '../../../hooks/useDebouncedValue'
import { useSearchMetadata } from '../hooks/useSearchMetadata'
import { RuleFiltersPanel } from '../../../components/RuleFiltersPanel'

export const SearchPage = () => {
  const { request, dateBounds, setFilterState, resetFilters } = useSearchFilters()
  const { data, isPending, isFetching, error } = useSearchRules(request)
  const metadataQuery = useSearchMetadata()

  const agencyOptions = useMemo(
    () => metadataQuery.data?.agencies ?? [],
    [metadataQuery.data?.agencies],
  )

  const rfiRfcOptions = useMemo(
    () => metadataQuery.data?.rfiRfcLabels ?? [],
    [metadataQuery.data?.rfiRfcLabels],
  )

  const [queryInput, setQueryInput] = useState(request.query ?? '')
  const debouncedQuery = useDebouncedValue(queryInput, 400)

  useEffect(() => {
    setQueryInput(request.query ?? '')
  }, [request.query])

  useEffect(() => {
    if (debouncedQuery !== (request.query ?? '')) {
      setFilterState({ query: debouncedQuery || undefined, page: 1 })
    }
  }, [debouncedQuery, request.query, setFilterState])

  return (
    <section className="space-y-6">
      <div className="card space-y-4 p-6">
        <div>
          <h2 className="text-2xl font-semibold text-slate-900">Search open comment periods</h2>
          <p className="mt-1 text-sm text-slate-600">
            Filter by agency and comment window to focus on regulations that align with your priorities.
          </p>
        </div>
        <div>
          <label className="flex flex-col gap-2 text-sm text-slate-600">
            <span className="font-medium text-slate-700">Search for regulations</span>
            <input
              type="search"
              value={queryInput}
              onChange={(event) => setQueryInput(event.target.value)}
              placeholder="i.e. PFAS, telehealth reimbursement, autonomous vehicles"
              className="w-full rounded-lg border border-slate-300 px-4 py-2.5 text-base shadow-sm focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-200"
            />
          </label>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-[280px_1fr]">
        <RuleFiltersPanel
          activeOnly={Boolean(request.activeOnly)}
          onActiveOnlyChange={(value) => setFilterState({ activeOnly: value, page: 1 })}
          agencyOptions={agencyOptions}
          selectedAgencies={request.agencies ?? []}
          rfiRfcOptions={rfiRfcOptions}
          selectedRfiRfcLabels={request.rfiRfcLabels ?? []}
          closingAfter={request.closingAfter}
          closingBefore={request.closingBefore}
          minDate={dateBounds.min}
          maxDate={dateBounds.max}
          onAgenciesChange={(values) => setFilterState({ agencies: values, page: 1 })}
          onRfiRfcChange={(values) => setFilterState({ rfiRfcLabels: values, page: 1 })}
          onClosingRangeChange={(range) => setFilterState({ ...range, page: 1 })}
          onReset={() => {
            setFilterState({
              agencies: [],
              rfiRfcLabels: [],
              closingAfter: undefined,
              closingBefore: undefined,
              activeOnly: false,
              page: 1,
            })
            resetFilters()
          }}
        />
        <SearchResults
          data={data}
          isLoading={isPending}
          isFetching={isFetching}
          error={error as Error | null}
          onPageChange={(page) => setFilterState({ page })}
        />
      </div>

      {metadataQuery.isPending ? (
        <p className="text-xs text-slate-400">Loading agency filters…</p>
      ) : null}
    </section>
  )
}
