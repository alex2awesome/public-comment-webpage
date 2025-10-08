import { RuleCard } from '../../../components/RuleCard'
import type { Recommendation } from '../../../lib/types'

interface RecommendationFeedProps {
  recommendations: Recommendation[]
}

const priorityStyles: Record<Recommendation['priority'], string> = {
  High: 'bg-red-50 text-red-600 border-red-200',
  Medium: 'bg-amber-50 text-amber-700 border-amber-200',
  Low: 'bg-emerald-50 text-emerald-600 border-emerald-200',
}

export const RecommendationFeed = ({ recommendations }: RecommendationFeedProps) => {
  return (
    <div className="flex flex-col gap-4">
      {recommendations.map((recommendation) => (
        <div key={recommendation.rule_id} className="card space-y-4 p-6">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <span className={`rounded-full border px-3 py-1 text-xs font-semibold ${priorityStyles[recommendation.priority]}`}>
                {recommendation.priority} priority
              </span>
              <span className="text-xs font-medium uppercase tracking-wide text-slate-500">
                Match score: {(recommendation.score * 100).toFixed(0)}%
              </span>
            </div>
            <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
              {recommendation.match_topics.map((topic) => (
                <span key={topic} className="rounded-full bg-brand-50 px-2.5 py-1 text-brand-600">
                  {topic}
                </span>
              ))}
            </div>
          </div>
          <RuleCard rule={recommendation.rule} />
        </div>
      ))}
    </div>
  )
}
