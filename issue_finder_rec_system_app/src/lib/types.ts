export type RuleStage =
  | 'Proposed Rule'
  | 'Final Rule'
  | 'Request for Information'
  | 'Notice'
  | 'Rule'
  | 'Guidance'
  | 'Other'
  | string

export interface Rule {
  id: string
  title: string
  type: string
  agency: string
  agency_acronym: string
  docket_id: string
  publication_date?: string
  comment_start_date?: string
  comment_due_date?: string
  comment_open_date: string
  comment_close_date: string
  comment_status?: string
  comment_active: boolean
  is_rfi_rfc: boolean
  rfi_rfc_label?: string
  summary: string
  abstract?: string
  details?: string
  supplementary_information?: string
  fr_document_number?: string
  topics: string[]
  comment_url: string
  source_url: string
  stage: RuleStage
}

export interface SearchRequest {
  query?: string
  agencies?: string[]
  rfiRfcLabels?: string[]
  topics?: string[]
  closingBefore?: string
  closingAfter?: string
  page?: number
  pageSize?: number
  activeOnly?: boolean
}

export interface SearchResponse {
  items: Rule[]
  total: number
  page: number
  pageSize: number
  totalPages: number
  appliedFilters: {
    agencies: string[]
    rfiRfcLabels: string[]
    topics: string[]
    closingBefore?: string
    closingAfter?: string
    activeOnly?: boolean
  }
}

export interface Recommendation {
  user_id: string
  rule_id: string
  score: number
  reasons: string[]
  match_topics: string[]
  priority: 'High' | 'Medium' | 'Low'
  rule: Rule
}

export interface SearchMetadata {
  agencies: Array<{ value: string; label: string }>
  rfiRfcLabels: Array<{ value: string; label: string }>
  topics: string[]
}

export interface UserProfile {
  id: string
  name: string
  role: string
  organization: string
  interests: string[]
  homepage_url: string
  last_updated: string
  summary?: string
}
