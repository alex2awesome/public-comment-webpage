import { Link, NavLink, Outlet, useLocation } from 'react-router-dom'

const navItems = [
  { to: '/search', label: 'Search' },
  { to: '/dashboard', label: 'Dashboard' },
]

export const AppShell = () => {
  const location = useLocation()
  const isHome = location.pathname === '/'

  return (
    <div className="min-h-screen bg-slate-50">
      <header className="border-b border-slate-200 bg-white/80 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-4">
          <Link to="/" className="flex items-center gap-2">
            <span className="rounded-md bg-brand-100 px-2 py-1 text-sm font-semibold text-brand-700">
              RegComments
            </span>
          </Link>
          <nav className="flex items-center gap-6 text-sm font-medium text-slate-600">
            {navItems.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) =>
                  `relative transition-colors hover:text-brand-600 ${isActive ? 'text-brand-600' : ''}`
                }
              >
                {({ isActive }) => (
                  <span>
                    {item.label}
                    {isActive ? (
                      <span className="absolute -bottom-2 left-0 right-0 h-0.5 rounded-full bg-brand-500" />
                    ) : null}
                  </span>
                )}
              </NavLink>
            ))}
          </nav>
          <div className="flex items-center gap-2">
            <button className="rounded-lg border border-slate-200 px-3 py-1.5 text-sm font-medium text-slate-600 hover:border-brand-400 hover:text-brand-600">
              Sign in
            </button>
          </div>
        </div>
      </header>
      <main className={`mx-auto w-full max-w-6xl px-4 py-8 ${isHome ? 'py-10' : 'py-8'}`}>
        <Outlet />
      </main>
      <footer className="border-t border-slate-200 bg-white py-6">
        <div className="mx-auto flex max-w-6xl flex-col gap-2 px-4 text-sm text-slate-500 sm:flex-row sm:items-center sm:justify-between">
          <p>&copy; {new Date().getFullYear()} RegComments. All rights reserved.</p>
          <div className="flex gap-4">
            <Link to="/privacy" className="hover:text-brand-600">
              Privacy
            </Link>
            <Link to="/about" className="hover:text-brand-600">
              About
            </Link>
          </div>
        </div>
      </footer>
    </div>
  )
}
