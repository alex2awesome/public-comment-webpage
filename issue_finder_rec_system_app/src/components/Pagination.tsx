interface PaginationProps {
  page: number
  totalPages: number
  onChange: (page: number) => void
  isLoading?: boolean
}

export const Pagination = ({ page, totalPages, onChange, isLoading = false }: PaginationProps) => {
  if (totalPages <= 1) return null

  const canGoBack = page > 1
  const canGoForward = page < totalPages

  const goToPage = (nextPage: number) => {
    if (nextPage === page || nextPage < 1 || nextPage > totalPages) return
    onChange(nextPage)
  }

  return (
    <div className="flex items-center justify-between border-t border-slate-200 pt-4 text-sm text-slate-600">
      <div>
        Page <span className="font-semibold text-slate-900">{page}</span> of{' '}
        <span className="font-semibold text-slate-900">{totalPages}</span>
      </div>
      <div className="flex items-center gap-2">
        <button
          onClick={() => goToPage(page - 1)}
          disabled={!canGoBack || isLoading}
          className="rounded-lg border border-slate-300 px-3 py-1.5 font-medium text-slate-600 transition hover:border-brand-400 hover:text-brand-600 disabled:cursor-not-allowed disabled:opacity-40"
        >
          Previous
        </button>
        <button
          onClick={() => goToPage(page + 1)}
          disabled={!canGoForward || isLoading}
          className="rounded-lg border border-slate-300 px-3 py-1.5 font-medium text-slate-600 transition hover:border-brand-400 hover:text-brand-600 disabled:cursor-not-allowed disabled:opacity-40"
        >
          Next
        </button>
      </div>
    </div>
  )
}
