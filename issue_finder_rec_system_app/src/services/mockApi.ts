import { loadRulesFromCsv, loadSearchMetadata } from './csvData'
import recommendationsFixture from '../fixtures/recommendations.json'
import userProfilesFixture from '../fixtures/userProfiles.json'
import type { Recommendation, Rule, SearchRequest, SearchResponse, UserProfile } from '../lib/types'
import { daysUntil } from '../lib/date'
import {
  isCommentPeriodActive,
  matchesAgencyFilter,
  matchesRfiRfcFilter,
  isWithinClosingRange,
} from '../lib/rules'

const artificialDelay = (ms = 200) => new Promise((resolve) => setTimeout(resolve, ms))

const normalize = (value: string) => value.toLowerCase()

const matchesQuery = (rule: Rule, query?: string) => {
  if (!query) return true
  const needle = normalize(query)
  return [rule.title, rule.summary, rule.topics.join(' '), rule.agency].some((field) =>
    normalize(field).includes(needle),
  )
}

const computeRelevanceScore = (rule: Rule, queryTokens: string[]) => {
  if (queryTokens.length === 0) {
    const dueIn = daysUntil(rule.comment_due_date ?? rule.comment_close_date)
    if (typeof dueIn === 'number' && dueIn > 0) {
      return 1 / Math.max(dueIn, 1)
    }
    return 0
  }

  const haystacks = [
    rule.title,
    rule.details ?? '',
    rule.abstract ?? '',
    rule.agency,
    rule.type ?? '',
    rule.comment_status ?? '',
    rule.fr_document_number ?? '',
  ].map((value) => value.toLowerCase())

  let score = 0
  for (const token of queryTokens) {
    const tokenScore = haystacks.reduce((acc, haystack) => {
      if (!token) return acc
      return haystack.includes(token) ? acc + 1 : acc
    }, 0)
    score += tokenScore
  }

  // Boost matches in title and agency slightly
  for (const token of queryTokens) {
    if (rule.title.toLowerCase().includes(token)) score += 2
    if (rule.agency.toLowerCase().includes(token)) score += 1
  }

  return score
}

const paginate = (items: Rule[], page = 1, pageSize = 10) => {
  const start = (page - 1) * pageSize
  const end = start + pageSize
  const sliced = items.slice(start, end)
  const totalPages = Math.max(1, Math.ceil(items.length / pageSize))

  return {
    items: sliced,
    total: items.length,
    page,
    pageSize,
    totalPages,
  }
}

const userProfiles = userProfilesFixture as UserProfile[]
const userProfileMap = new Map(userProfiles.map((profile) => [profile.id, profile]))

type RecommendationEntry = {
  rule_id: string
  score: number
  priority?: Recommendation['priority']
  match_topics?: string[]
  reasons?: string[]
}

const recommendationMap = new Map<string, RecommendationEntry[]>(
  (recommendationsFixture as Array<{ user_id: string; recommendations: RecommendationEntry[] }>).map((entry) => [
    entry.user_id,
    entry.recommendations ?? [],
  ]),
)

export const mockApi = {
  async searchRules(request: SearchRequest): Promise<SearchResponse> {
    await artificialDelay()
    const rules = await loadRulesFromCsv()

    const {
      query,
      agencies = [],
      closingAfter,
      closingBefore,
      activeOnly = false,
      rfiRfcLabels = [],
      page = 1,
      pageSize = 10,
    } = request

    const tokens = query
      ? query
          .toLowerCase()
          .split(/\s+/)
          .map((token) => token.trim())
          .filter(Boolean)
      : []

    const filteredWithScore = rules
      .filter((rule) => matchesQuery(rule, query))
      .filter((rule) => matchesAgencyFilter(rule, agencies))
      .filter((rule) => matchesRfiRfcFilter(rule, rfiRfcLabels))
      .filter((rule) => isWithinClosingRange(rule, closingAfter, closingBefore))
      .filter((rule) => (activeOnly ? isCommentPeriodActive(rule) : true))
      .map((rule) => ({
        rule,
        score: computeRelevanceScore(rule, tokens),
      }))
      .sort((a, b) => {
        if (a.rule.comment_active !== b.rule.comment_active) {
          return a.rule.comment_active ? -1 : 1
        }
        if (b.score !== a.score) {
          return b.score - a.score
        }
        const dueA = new Date(a.rule.comment_due_date ?? a.rule.comment_close_date).getTime()
        const dueB = new Date(b.rule.comment_due_date ?? b.rule.comment_close_date).getTime()
        return dueA - dueB
      })

    const sortedRules = filteredWithScore.map((entry) => entry.rule)

    const pagination = paginate(sortedRules, page, pageSize)

    return {
      ...pagination,
      appliedFilters: {
        agencies,
        rfiRfcLabels,
        topics: [],
        closingAfter,
        closingBefore,
        activeOnly,
      },
    }
  },

  async getRule(ruleId: string): Promise<Rule | undefined> {
    await artificialDelay()
    const rules = await loadRulesFromCsv()
    return rules.find((rule) => rule.id === ruleId)
  },

  async getRelatedRules(ruleId: string): Promise<Rule[]> {
    await artificialDelay()
    const rules = await loadRulesFromCsv()
    const target = rules.find((rule) => rule.id === ruleId)
    if (!target) return []

    const targetTopics = new Set(target.topics.map(normalize))
    return rules
      .filter((rule) => rule.id !== ruleId)
      .map((rule) => ({
        rule,
        overlap: rule.topics.reduce((acc, topic) => (targetTopics.has(normalize(topic)) ? acc + 1 : acc), 0),
      }))
      .filter(({ overlap }) => overlap > 0)
      .sort((a, b) => b.overlap - a.overlap)
      .slice(0, 3)
      .map(({ rule }) => rule)
  },

  async getRecommendations(userId: string): Promise<Recommendation[]> {
    const rules = await loadRulesFromCsv()
    const ruleLookup = new Map(rules.map((rule) => [rule.id, rule]))
    const entries = recommendationMap.get(userId) ?? []

    return entries
      .map((entry) => {
        const rule = ruleLookup.get(entry.rule_id)
        if (!rule) return null
        return {
          user_id: userId,
          rule_id: entry.rule_id,
          score: Number(entry.score ?? 0),
          reasons: entry.reasons ?? [],
          match_topics: entry.match_topics ?? [],
          priority: entry.priority ?? 'Low',
          rule,
        } as Recommendation
      })
      .filter((value): value is Recommendation => Boolean(value))
  },

  async getUserProfile(userId: string): Promise<UserProfile | undefined> {
    await artificialDelay()
    return userProfileMap.get(userId)
  },

  async listUserProfiles(): Promise<UserProfile[]> {
    await artificialDelay()
    return userProfiles
  },

  async getSearchMetadata() {
    await artificialDelay()
    return loadSearchMetadata()
  },
}
