import { useEffect, useMemo, useState } from 'react'
import { RecommendationFeed } from '../components/RecommendationFeed'
import { useRecommendationProfiles, useRecommendations } from '../hooks/useRecommendations'
import { formatDate } from '../../../lib/date'
import { RuleFiltersPanel } from '../../../components/RuleFiltersPanel'
import {
  isCommentPeriodActive,
  isWithinClosingRange,
  matchesAgencyFilter,
  matchesRfiRfcFilter,
} from '../../../lib/rules'
import type { Recommendation } from '../../../lib/types'

export const DashboardPage = () => {
  const { data: collaborators, isPending: isProfilesLoading, error: profilesError } = useRecommendationProfiles()
  const [selectedUserId, setSelectedUserId] = useState<string | undefined>(undefined)
  const [showActiveOnly, setShowActiveOnly] = useState(false)
  const [isEditingProfile, setIsEditingProfile] = useState(false)
  const [editableProfileText, setEditableProfileText] = useState('')
  const [closingAfter, setClosingAfter] = useState<string | undefined>(undefined)
  const [closingBefore, setClosingBefore] = useState<string | undefined>(undefined)
  const [selectedAgencies, setSelectedAgencies] = useState<string[]>([])
  const [selectedRfiRfcLabels, setSelectedRfiRfcLabels] = useState<string[]>([])

  useEffect(() => {
    if (!selectedUserId && collaborators && collaborators.length > 0) {
      setSelectedUserId(collaborators[0].id)
    }
  }, [collaborators, selectedUserId])

  useEffect(() => {
    setShowActiveOnly(false)
    setClosingAfter(undefined)
    setClosingBefore(undefined)
    setSelectedAgencies([])
    setSelectedRfiRfcLabels([])
  }, [selectedUserId])

  const {
    user,
    recommendations,
    isUserLoading,
    isRecommendationsLoading,
    isFetching,
    error: recommendationError,
  } = useRecommendations(selectedUserId)

  const error = recommendationError ?? profilesError

  const showLoadingState = isProfilesLoading || isUserLoading

  const activeRecommendationCount = useMemo(
    () => (recommendations ?? []).filter((item) => isCommentPeriodActive(item.rule)).length,
    [recommendations],
  )

  const closedRecommendationCount = (recommendations?.length ?? 0) - activeRecommendationCount

  const agencyOptions = useMemo(() => {
    if (!recommendations) return []

    const seen = new Map<string, string>()

    recommendations.forEach((item) => {
      const acronym = item.rule.agency_acronym?.trim()
      const fullName = item.rule.agency?.trim()
      const value = acronym || fullName
      if (!value || seen.has(value)) return

      const label = acronym && fullName && acronym !== fullName ? `${acronym} · ${fullName}` : fullName || acronym
      seen.set(value, label)
    })

    return Array.from(seen.entries())
      .map(([value, label]) => ({ value, label }))
      .sort((a, b) => a.label.localeCompare(b.label))
  }, [recommendations])

  const rfiRfcOptions = useMemo(() => {
    if (!recommendations) return []

    const seen = new Set<string>()
    recommendations.forEach((item) => {
      if (item.rule.is_rfi_rfc && item.rule.rfi_rfc_label) {
        seen.add(item.rule.rfi_rfc_label)
      }
    })

    return Array.from(seen)
      .map((label) => ({ value: label, label }))
      .sort((a, b) => a.label.localeCompare(b.label))
  }, [recommendations])

  const closingDateBounds = useMemo(() => {
    if (!recommendations || recommendations.length === 0) {
      return {
        min: '2000-01-01',
        max: '2100-12-31',
      }
    }

    const timestamps: number[] = []
    recommendations.forEach((item) => {
      const candidate = item.rule.comment_close_date ?? item.rule.comment_due_date
      if (candidate) {
        const parsed = new Date(candidate)
        if (!Number.isNaN(parsed.getTime())) {
          timestamps.push(parsed.getTime())
        }
      }
    })

    if (timestamps.length === 0) {
      return {
        min: '2000-01-01',
        max: '2100-12-31',
      }
    }

    const toIso = (date: Date) => date.toISOString().slice(0, 10)

    return {
      min: toIso(new Date(Math.min(...timestamps))),
      max: toIso(new Date(Math.max(...timestamps))),
    }
  }, [recommendations])

  const filteredRecommendations = useMemo(() => {
    if (!recommendations) return []

    const matchesFilters = (item: Recommendation) => {
      const { rule } = item
      if (showActiveOnly && !isCommentPeriodActive(rule)) return false
      if (!isWithinClosingRange(rule, closingAfter, closingBefore)) return false
      if (!matchesAgencyFilter(rule, selectedAgencies)) return false
      if (!matchesRfiRfcFilter(rule, selectedRfiRfcLabels)) return false
      return true
    }

    return recommendations.filter(matchesFilters)
  }, [
    recommendations,
    showActiveOnly,
    closingAfter,
    closingBefore,
    selectedAgencies,
    selectedRfiRfcLabels,
  ])

  const formattedProfile = useMemo(() => {
    if (!user) return ''

    const interests = user.interests.length ? `Interests: ${user.interests.join(', ')}` : undefined
    const summary = user.summary?.trim() ? `Summary: ${user.summary.trim()}` : undefined

    return [user.name, user.role, user.organization, interests, summary]
      .filter(Boolean)
      .join('\n')
  }, [user])

  useEffect(() => {
    if (isEditingProfile) {
      setEditableProfileText(formattedProfile)
    }
  }, [formattedProfile, isEditingProfile])

  if (error) {
    return (
      <div className="card space-y-3 p-8 text-center">
        <h2 className="text-2xl font-semibold text-red-600">Unable to load recommendations</h2>
        <p className="text-sm text-slate-600">{(error as Error).message}</p>
      </div>
    )
  }

  return (
    <section className="space-y-6">
      <header className="card space-y-4 p-6">
        <div className="flex flex-col gap-2">
          <p className="text-xs font-semibold uppercase tracking-wide text-brand-600">Personalized overview</p>
          <h2 className="text-2xl font-semibold text-slate-900">Recommendations dashboard</h2>
        </div>

        {collaborators && collaborators.length > 0 ? (
          <div className="flex flex-wrap items-center gap-2">
            {collaborators.map((profile) => {
              const isActive = profile.id === selectedUserId
              return (
                <button
                  key={profile.id}
                  onClick={() => setSelectedUserId(profile.id)}
                  className={`rounded-full border px-3 py-1.5 text-sm font-medium transition ${
                    isActive
                      ? 'border-brand-500 bg-brand-50 text-brand-700'
                      : 'border-slate-200 bg-white text-slate-600 hover:border-brand-300 hover:text-brand-600'
                  }`}
                  type="button"
                >
                  {profile.name}
                </button>
              )
            })}
          </div>
        ) : null}

        {user ? (
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="text-sm font-medium text-slate-700">{user.name}</p>
              <p className="text-sm text-slate-600">
                {user.role}
                {user.organization ? ` · ${user.organization}` : ''}
              </p>
            </div>
            <div className="flex flex-col items-start gap-3 text-xs text-slate-500 sm:flex-row sm:items-center">
              <div className="flex flex-wrap items-center gap-3 text-xs text-slate-500">
                <span>
                  Profile updated{' '}
                  <strong className="text-slate-700">{formatDate(user.last_updated)}</strong>
                </span>
                {user.interests.length ? (
                  <>
                    <span className="hidden h-4 w-px bg-slate-200 sm:block" />
                    <span>Focus areas: {user.interests.join(', ')}</span>
                  </>
                ) : null}
              </div>
              <button
                type="button"
                onClick={() => setIsEditingProfile((previous) => !previous)}
                className="w-full rounded-lg border border-slate-200 px-3 py-1.5 text-sm font-medium text-slate-600 transition hover:border-brand-400 hover:text-brand-600 sm:w-auto"
              >
                {isEditingProfile ? 'Close profile editor' : 'Edit profile'}
              </button>
            </div>
          </div>
        ) : showLoadingState ? (
          <div className="h-14 animate-pulse rounded-lg bg-slate-200/70" />
        ) : null}

        {isEditingProfile && user ? (
          <div className="space-y-2 rounded-lg border border-slate-200 bg-slate-50 p-4">
            <label className="flex flex-col gap-2 text-sm text-slate-600">
              <span className="font-medium text-slate-700">{user.name}'s profile</span>
              <textarea
                value={editableProfileText}
                onChange={(event) => setEditableProfileText(event.target.value)}
                rows={12}
                className="min-h-[180px] w-full rounded-lg border border-slate-300 px-4 py-3 text-sm shadow-sm focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-200"
              />
            </label>
            <p className="text-xs text-slate-500">Editable snapshot of this profile as currently stored.</p>
          </div>
        ) : null}
      </header>

      <div className="grid gap-6 lg:grid-cols-[280px_1fr]">
        <RuleFiltersPanel
          activeOnly={showActiveOnly}
          onActiveOnlyChange={setShowActiveOnly}
          closingAfter={closingAfter}
          closingBefore={closingBefore}
          minDate={closingDateBounds.min}
          maxDate={closingDateBounds.max}
          onClosingRangeChange={({ closingAfter: after, closingBefore: before }) => {
            setClosingAfter(after)
            setClosingBefore(before)
          }}
          selectedAgencies={selectedAgencies}
          agencyOptions={agencyOptions}
          onAgenciesChange={setSelectedAgencies}
          selectedRfiRfcLabels={selectedRfiRfcLabels}
          rfiRfcOptions={rfiRfcOptions}
          onRfiRfcChange={setSelectedRfiRfcLabels}
          onReset={() => {
            setShowActiveOnly(false)
            setClosingAfter(undefined)
            setClosingBefore(undefined)
            setSelectedAgencies([])
            setSelectedRfiRfcLabels([])
          }}
          disabled={isRecommendationsLoading && !recommendations}
          stats={{
            activeCount: activeRecommendationCount,
            closedCount: closedRecommendationCount,
          }}
        />

        {isRecommendationsLoading || !recommendations ? (
          <div className="card space-y-3 p-8">
            <div className="h-4 animate-pulse rounded bg-slate-200/70" />
            <div className="h-4 animate-pulse rounded bg-slate-200/70" />
            <div className="h-4 animate-pulse rounded bg-slate-200/70" />
          </div>
        ) : filteredRecommendations.length > 0 ? (
          <div className="space-y-3">
            {isFetching ? <p className="text-xs text-slate-400">Refreshing recommendations…</p> : null}
            <RecommendationFeed recommendations={filteredRecommendations} />
          </div>
        ) : (
          <div className="card space-y-3 p-10 text-center">
            <h3 className="text-lg font-semibold text-slate-800">No recommendations match the current filters</h3>
            <p className="text-sm text-slate-600">
              {showActiveOnly
                ? 'All current recommendations have closed comment periods. Clear the filter to see them.'
                : 'As soon as similarity scores surface relevant comment periods for the selected collaborator, they will appear here.'}
            </p>
          </div>
        )}
      </div>
    </section>
  )
}
