import { useCallback, type ChangeEvent, type FC } from 'react'

interface Option {
  label: string
  value: string
}

interface RuleFiltersPanelProps {
  activeOnly: boolean
  onActiveOnlyChange: (value: boolean) => void
  closingAfter?: string
  closingBefore?: string
  minDate: string
  maxDate: string
  onClosingRangeChange: (range: { closingAfter?: string; closingBefore?: string }) => void
  selectedAgencies: string[]
  agencyOptions: Option[]
  onAgenciesChange: (values: string[]) => void
  selectedRfiRfcLabels: string[]
  rfiRfcOptions: Option[]
  onRfiRfcChange: (values: string[]) => void
  onReset: () => void
  disabled?: boolean
  stats?: {
    activeCount: number
    closedCount: number
  }
}

const toggleValue = (values: string[], nextValue: string) =>
  values.includes(nextValue) ? values.filter((value) => value !== nextValue) : [...values, nextValue]

export const RuleFiltersPanel: FC<RuleFiltersPanelProps> = ({
  activeOnly,
  onActiveOnlyChange,
  closingAfter,
  closingBefore,
  minDate,
  maxDate,
  onClosingRangeChange,
  selectedAgencies,
  agencyOptions,
  onAgenciesChange,
  selectedRfiRfcLabels,
  rfiRfcOptions,
  onRfiRfcChange,
  onReset,
  disabled = false,
  stats,
}) => {
  const canReset =
    activeOnly ||
    Boolean(closingAfter) ||
    Boolean(closingBefore) ||
    selectedAgencies.length > 0 ||
    selectedRfiRfcLabels.length > 0

  const handleAgencyChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const { value } = event.target
      onAgenciesChange(toggleValue(selectedAgencies, value))
    },
    [onAgenciesChange, selectedAgencies],
  )

  const handleRfiRfcChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const { value } = event.target
      onRfiRfcChange(toggleValue(selectedRfiRfcLabels, value))
    },
    [onRfiRfcChange, selectedRfiRfcLabels],
  )

  const handleClosingAfterChange = (event: ChangeEvent<HTMLInputElement>) => {
    const value = event.target.value || undefined
    onClosingRangeChange({ closingAfter: value, closingBefore })
  }

  const handleClosingBeforeChange = (event: ChangeEvent<HTMLInputElement>) => {
    const value = event.target.value || undefined
    onClosingRangeChange({ closingAfter, closingBefore: value })
  }

  return (
    <aside className="card h-fit space-y-6 p-6">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-500">Filters</h3>
        <button
          type="button"
          onClick={onReset}
          disabled={!canReset || disabled}
          className={`text-sm font-medium transition ${
            canReset && !disabled
              ? 'text-brand-600 hover:text-brand-500'
              : 'cursor-not-allowed text-slate-300'
          }`}
        >
          Clear all
        </button>
      </div>

      <div className="space-y-3 text-sm text-slate-600">
        <div className="rounded-lg border border-slate-200 px-4 py-3 shadow-sm">
          <label className="flex items-start justify-between gap-3">
            <span className="flex-1">
              <span className="block text-sm font-semibold text-slate-700">Active comment periods</span>
              <span className="mt-1 block text-xs text-slate-500">
                Show only regulations where public comments are still open.
              </span>
            </span>
            <input
              type="checkbox"
              checked={activeOnly}
              onChange={(event) => onActiveOnlyChange(event.target.checked)}
              disabled={disabled}
              className="mt-1 h-4 w-4 rounded border-slate-300 text-brand-600 focus:ring-brand-500"
            />
          </label>
        </div>

        {stats ? (
          <div className="rounded-lg bg-slate-100 px-4 py-3 text-xs text-slate-500">
            <p>
              {stats.activeCount} active {stats.activeCount === 1 ? 'rule' : 'rules'} · {stats.closedCount} closed
            </p>
          </div>
        ) : null}
      </div>

      <div className="space-y-2">
        <h4 className="text-sm font-semibold text-slate-700">Comment window</h4>
        <label className="flex flex-col gap-1 text-sm text-slate-600">
          <span>Closing after</span>
          <input
            type="date"
            value={closingAfter ?? ''}
            onChange={handleClosingAfterChange}
            min={minDate}
            max={maxDate}
            disabled={disabled}
            className="rounded-lg border border-slate-300 px-3 py-2 text-sm shadow-sm focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-200 disabled:cursor-not-allowed disabled:bg-slate-100"
          />
        </label>
        <label className="flex flex-col gap-1 text-sm text-slate-600">
          <span>Closing before</span>
          <input
            type="date"
            value={closingBefore ?? ''}
            onChange={handleClosingBeforeChange}
            min={minDate}
            max={maxDate}
            disabled={disabled}
            className="rounded-lg border border-slate-300 px-3 py-2 text-sm shadow-sm focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-200 disabled:cursor-not-allowed disabled:bg-slate-100"
          />
        </label>
      </div>

      <div className="space-y-2">
        <h4 className="text-sm font-semibold text-slate-700">Agencies</h4>
        <div className="mt-2 grid gap-2">
          {agencyOptions.length === 0 ? (
            <span className="text-xs text-slate-400">No agencies available.</span>
          ) : (
            agencyOptions.map((option) => (
              <label key={option.value} className="inline-flex items-center gap-2 text-sm text-slate-600">
                <input
                  type="checkbox"
                  value={option.value}
                  checked={selectedAgencies.includes(option.value)}
                  onChange={handleAgencyChange}
                  disabled={disabled}
                  className="h-4 w-4 rounded border-slate-300 text-brand-600 focus:ring-brand-500"
                />
                {option.label}
              </label>
            ))
          )}
        </div>
      </div>

      <div className="space-y-2">
        <h4 className="text-sm font-semibold text-slate-700">RFI / RFC</h4>
        <div className="mt-2 grid gap-2">
          {rfiRfcOptions.length === 0 ? (
            <span className="text-xs text-slate-400">No RFI or RFC labels detected.</span>
          ) : (
            rfiRfcOptions.map((option) => (
              <label key={option.value} className="inline-flex items-center gap-2 text-sm text-slate-600">
                <input
                  type="checkbox"
                  value={option.value}
                  checked={selectedRfiRfcLabels.includes(option.value)}
                  onChange={handleRfiRfcChange}
                  disabled={disabled}
                  className="h-4 w-4 rounded border-slate-300 text-brand-600 focus:ring-brand-500"
                />
                {option.label}
              </label>
            ))
          )}
        </div>
      </div>
    </aside>
  )
}
