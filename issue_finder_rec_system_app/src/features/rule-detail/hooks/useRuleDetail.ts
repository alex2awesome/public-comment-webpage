import { useQuery } from '@tanstack/react-query'
import { fetchRelatedRules, fetchRuleDetail } from '../../../services/rules.service'

export const useRuleDetail = (ruleId: string | undefined) => {
  const ruleQuery = useQuery({
    queryKey: ['rule-detail', ruleId],
    queryFn: () => (ruleId ? fetchRuleDetail(ruleId) : Promise.resolve(undefined)),
    enabled: Boolean(ruleId),
  })

  const relatedQuery = useQuery({
    queryKey: ['rule-related', ruleId],
    queryFn: () => (ruleId ? fetchRelatedRules(ruleId) : Promise.resolve([])),
    enabled: Boolean(ruleId),
  })

  return {
    rule: ruleQuery.data,
    isLoading: ruleQuery.isPending,
    error: ruleQuery.error,
    related: relatedQuery.data ?? [],
    isLoadingRelated: relatedQuery.isPending,
  }
}
