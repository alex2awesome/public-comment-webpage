import { createBrowserRouter } from 'react-router-dom'
import { AppShell } from '../components/AppShell'
import { HomePage } from '../features/home/pages/HomePage'
import { SearchPage } from '../features/search/pages/SearchPage'
import { DashboardPage } from '../features/recommendations/pages/DashboardPage'
import { RuleDetailPage } from '../features/rule-detail/pages/RuleDetailPage'
import { NotFoundPage } from './NotFoundPage'

const basename = import.meta.env.BASE_URL.replace(/\/$/, '') || '/'

export const router = createBrowserRouter(
  [
    {
      path: '/',
      element: <AppShell />,
      children: [
        { index: true, element: <HomePage /> },
        { path: 'search', element: <SearchPage /> },
        { path: 'rules/:ruleId', element: <RuleDetailPage /> },
        { path: 'dashboard', element: <DashboardPage /> },
        { path: '*', element: <NotFoundPage /> },
      ],
    },
  ],
  {
    basename,
  },
)
