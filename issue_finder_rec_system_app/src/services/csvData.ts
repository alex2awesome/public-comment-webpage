import { csvParse } from 'd3-dsv'
import type { Rule } from '../lib/types'

const csvModules = import.meta.glob('@data/*.csv', { eager: true, import: 'default', query: '?raw' }) as Record<string, string>

const unique = <T>(values: T[]) => Array.from(new Set(values))

const normalizeWhitespace = (value: string | undefined | null) =>
  (value ?? '')
    .replace(/\s+/g, ' ')
    .trim()

const parseBooleanFlag = (value?: string | null) => {
  const normalized = normalizeWhitespace(value)?.toLowerCase()
  if (!normalized) return false
  return normalized === 'true' || normalized === '1' || normalized === 'yes'
}

const formatRfiRfcLabel = (value?: string | null) => {
  const normalized = normalizeWhitespace(value)
  if (!normalized) return undefined
  const lower = normalized.toLowerCase()
  if (lower === 'none' || lower === 'false' || lower === 'n/a') return undefined
  return normalized
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase())
}

const normalizeDate = (value?: string) => {
  if (!value) return undefined
  const trimmed = value.trim()
  if (!trimmed) return undefined
  if (/^\d{4}-\d{2}-\d{2}$/.test(trimmed)) return trimmed
  const parsed = new Date(trimmed)
  if (Number.isNaN(parsed.getTime())) return undefined
  return parsed.toISOString().slice(0, 10)
}

const fallbackDate = (...values: Array<string | undefined>) => {
  for (const value of values) {
    const normalized = normalizeDate(value)
    if (normalized) return normalized
  }
  return new Date().toISOString().slice(0, 10)
}

const computeTopics = (details: string, type: string) => {
  const baseTokens = details
    .replace(/["“”]/g, '')
    .split(/[;:\-•]/)
    .map((token) => normalizeWhitespace(token))
    .filter(Boolean)

  const typed = normalizeWhitespace(type)
  return unique([...baseTokens.slice(0, 6), typed || 'Regulation']).filter(Boolean)
}

const computeAcronym = (agency: string) => {
  const words = agency.split(/[^A-Za-z0-9]+/).filter(Boolean)
  const uppercaseLetters = (agency.match(/[A-Z]/g) ?? []).join('')
  if (uppercaseLetters.length >= 2) return uppercaseLetters.slice(0, 6)
  if (words.length === 1) return words[0].slice(0, 4).toUpperCase()
  return words
    .slice(0, 5)
    .map((word) => word[0]?.toUpperCase() ?? '')
    .join('')
    .slice(0, 6)
}

const parseRow = (row: Record<string, string>, source: string): Rule | null => {
  const id = normalizeWhitespace(row.fr_document_number) || normalizeWhitespace(row.source) || `${source}-${row.title}`
  const title = normalizeWhitespace(row.title)
  const agency = normalizeWhitespace(row.agency) || 'Unknown Agency'
  if (!title) return null

  const publicationDate = normalizeDate(row.publication_date)
  const commentStartInput = normalizeDate(row.comment_start_date)
  const commentDueInput = normalizeDate(row.comment_due_date)
  const commentOpen = commentStartInput ?? fallbackDate(row.comment_start_date, row.publication_date)
  const commentClose = commentDueInput ?? fallbackDate(row.comment_due_date, row.comment_start_date, row.publication_date)
  const detailText = normalizeWhitespace(row.details) || title
  const abstract = normalizeWhitespace(row.abstract)
  const supplementary = normalizeWhitespace(row.supplementary_information)
  const commentStatus = normalizeWhitespace(row.comment_status)
  const commentActive = normalizeWhitespace(row.comment_active).toLowerCase()
  const isActive = commentActive === 'true' || commentActive === '1' || commentStatus.toLowerCase() === 'open'
  const topics = computeTopics(detailText, row.type || commentStatus || 'Regulation')
  const isRfiRfc = parseBooleanFlag(row.is_rfi_rfc)
  const rfiRfcLabel = isRfiRfc ? formatRfiRfcLabel(row.rfi_rfc_label) : undefined

  const rule: Rule = {
    id,
    title,
    type: normalizeWhitespace(row.type) || 'Notice',
    agency,
    agency_acronym: computeAcronym(agency),
    docket_id: normalizeWhitespace(row.docket_id) || 'N/A',
    publication_date: publicationDate,
    comment_start_date: commentStartInput ?? commentOpen,
    comment_due_date: commentDueInput ?? commentClose,
    comment_open_date: commentOpen,
    comment_close_date: commentClose,
    comment_status: commentStatus,
    comment_active: isActive,
    is_rfi_rfc: isRfiRfc,
    rfi_rfc_label: rfiRfcLabel,
    summary: abstract || detailText,
    abstract: abstract || undefined,
    details: detailText,
    supplementary_information: supplementary || undefined,
    fr_document_number: normalizeWhitespace(row.fr_document_number) || undefined,
    topics,
    comment_url: normalizeWhitespace(row.regs_url) || normalizeWhitespace(row.fr_url) || '#',
    source_url: normalizeWhitespace(row.fr_url) || normalizeWhitespace(row.regs_url) || '#',
    stage: normalizeWhitespace(row.type) || commentStatus || 'Notice',
  }

  return rule
}

let cachedRules: Rule[] | null = null

export const loadRulesFromCsv = async (): Promise<Rule[]> => {
  if (cachedRules) return cachedRules

  const aggregated = new Map<string, Rule>()

  for (const [path, rawCsv] of Object.entries(csvModules)) {
    const rows = csvParse(rawCsv)
    for (const row of rows) {
      const rule = parseRow(row as Record<string, string>, path)
      if (!rule) continue
      aggregated.set(rule.id, rule)
    }
  }

  const rules = Array.from(aggregated.values()).sort((a, b) =>
    a.comment_close_date.localeCompare(b.comment_close_date),
  )

  cachedRules = rules
  return rules
}

export const loadSearchMetadata = async () => {
  const rules = await loadRulesFromCsv()
  const agencyMap = new Map<string, { acronym?: string; name?: string }>()
  for (const rule of rules) {
    const acronym = rule.agency_acronym?.trim()
    const name = rule.agency?.trim()
    const value = acronym || name
    if (!value || agencyMap.has(value)) continue
    agencyMap.set(value, { acronym, name })
  }

  const agencies = Array.from(agencyMap.entries())
    .map(([value, { acronym, name }]) => {
      const hasBoth = Boolean(acronym && name && acronym !== name)
      const label = hasBoth ? `${acronym} · ${name}` : name || acronym || value
      return {
        value,
        label,
      }
    })
    .sort((a, b) => a.label.localeCompare(b.label))

  const topics = unique(rules.flatMap((rule) => rule.topics)).sort()
  const rfiRfcLabels = unique(
    rules
      .map((rule) => rule.rfi_rfc_label)
      .filter((label): label is string => Boolean(label)),
  )
    .map((label) => ({ value: label, label }))
    .sort((a, b) => a.label.localeCompare(b.label))

  return {
    agencies,
    rfiRfcLabels,
    topics,
  }
}
