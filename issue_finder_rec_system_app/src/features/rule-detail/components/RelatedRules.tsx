import { RuleCard } from '../../../components/RuleCard'
import type { Rule } from '../../../lib/types'

interface RelatedRulesProps {
  rules: Rule[]
}

export const RelatedRules = ({ rules }: RelatedRulesProps) => {
  if (!rules.length) return null

  return (
    <section className="space-y-3">
      <h3 className="text-lg font-semibold text-slate-900">Related regulations</h3>
      <p className="text-sm text-slate-600">
        These items share overlapping topics or agencies with the current regulation and may be relevant for the same
        reviewers.
      </p>
      <div className="flex flex-col gap-4">
        {rules.map((rule) => (
          <RuleCard key={rule.id} rule={rule} showActions={false} />
        ))}
      </div>
    </section>
  )
}
