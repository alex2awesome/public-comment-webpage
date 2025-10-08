import { keepPreviousData, useQuery } from '@tanstack/react-query'
import { searchRules } from '../../../services/rules.service'
import type { SearchRequest, SearchResponse } from '../../../lib/types'

export const useSearchRules = (request: SearchRequest) => {
  return useQuery<SearchResponse>({
    queryKey: ['rules', request],
    queryFn: () => searchRules(request),
    placeholderData: keepPreviousData,
  })
}
