import { Link } from 'react-router-dom'

export const HomePage = () => {
  return (
    <section className="flex flex-col gap-10">
      <div className="flex flex-col gap-6 text-center sm:gap-8">
        <div className="mx-auto max-w-2xl">
          <span className="rounded-full bg-brand-100 px-3 py-1 text-sm font-semibold text-brand-700">
            Your assistant for staying on top of federal rulemaking.
          </span>
          <h1 className="mt-4 text-4xl font-semibold text-slate-900 sm:text-5xl">
            Together, let's increase participation in our democracy.
          </h1>
          <p className="mt-4 text-lg text-slate-600">
            Search open comment periods, monitor deadlines, and surface the most relevant regulations for every expert on your team.
          </p>
        </div>
        <div className="mx-auto flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-center">
          <Link
            to="/search"
            className="inline-flex items-center justify-center rounded-lg bg-brand-600 px-6 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-brand-500"
          >
            Start searching
          </Link>
          <Link
            to="/dashboard"
            className="inline-flex items-center justify-center rounded-lg border border-slate-200 px-6 py-3 text-sm font-semibold text-slate-700 transition hover:border-brand-300 hover:text-brand-600"
          >
            View recommended rules
          </Link>
        </div>
      </div>
      <div className="grid gap-4 sm:grid-cols-3">
        cute picture here.
      </div>
    </section>
  )
}