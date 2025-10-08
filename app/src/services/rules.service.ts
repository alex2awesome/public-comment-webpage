import type { SearchMetadata, SearchRequest } from '../lib/types'
import { mockApi } from './mockApi'

export const searchRules = (params: SearchRequest) => mockApi.searchRules(params)
export const fetchRuleDetail = (ruleId: string) => mockApi.getRule(ruleId)
export const fetchRelatedRules = (ruleId: string) => mockApi.getRelatedRules(ruleId)
export const fetchSearchMetadata = (): Promise<SearchMetadata> => mockApi.getSearchMetadata()
