import { Link } from 'react-router-dom'

export const NotFoundPage = () => {
  return (
    <div className="mx-auto max-w-xl text-center">
      <h2 className="text-3xl font-semibold text-slate-900">Page not found</h2>
      <p className="mt-4 text-base text-slate-600">
        We couldn't find the page you're looking for. Try returning to search or the dashboard.
      </p>
      <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
        <Link to="/" className="rounded-lg bg-brand-600 px-4 py-2 text-sm font-semibold text-white hover:bg-brand-500">
          Go home
        </Link>
        <Link
          to="/search"
          className="rounded-lg border border-slate-200 px-4 py-2 text-sm font-semibold text-slate-700 hover:border-brand-300 hover:text-brand-600"
        >
          Search regulations
        </Link>
      </div>
    </div>
  )
}
