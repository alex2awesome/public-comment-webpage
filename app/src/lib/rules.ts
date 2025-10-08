import type { Rule } from './types'
import { daysUntil } from './date'

const normalize = (value?: string) => value?.trim().toLowerCase() ?? ''

export const isCommentPeriodActive = (rule: Rule) => {
  const daysRemaining = daysUntil(rule.comment_due_date ?? rule.comment_close_date)
  if (typeof daysRemaining === 'number') {
    return daysRemaining > 0
  }
  return rule.comment_active
}

export const isWithinClosingRange = (rule: Rule, after?: string, before?: string) => {
  const closingReference = rule.comment_close_date ?? rule.comment_due_date
  if (!closingReference) return true

  const closingDate = new Date(closingReference)
  if (Number.isNaN(closingDate.getTime())) return true

  if (after && closingDate < new Date(after)) return false
  if (before && closingDate > new Date(before)) return false
  return true
}

export const matchesAgencyFilter = (rule: Rule, agencies?: string[]) => {
  if (!agencies || agencies.length === 0) return true

  const ruleValues = [normalize(rule.agency_acronym), normalize(rule.agency)].filter(Boolean)
  if (ruleValues.length === 0) return true

  return agencies
    .map((value) => normalize(value))
    .filter(Boolean)
    .some((candidate) => ruleValues.includes(candidate))
}

export const matchesRfiRfcFilter = (rule: Rule, labels?: string[]) => {
  if (!labels || labels.length === 0) return true
  if (!rule.is_rfi_rfc) return false

  const ruleLabel = normalize(rule.rfi_rfc_label)
  if (!ruleLabel) return false

  return labels
    .map((label) => normalize(label))
    .filter(Boolean)
    .some((candidate) => candidate === ruleLabel)
}
