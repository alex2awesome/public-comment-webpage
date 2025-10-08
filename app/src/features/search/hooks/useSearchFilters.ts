import { useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'
import type { SearchRequest } from '../../../lib/types'

const parseList = (value: string | null) =>
  value?.split(',').map((item) => item.trim()).filter(Boolean) ?? []

const serializeList = (values?: string[]) => (values && values.length > 0 ? values.join(',') : null)

const MIN_YEAR = 2000
const MAX_YEAR = 2100

const parseDateParam = (value: string | null) => {
  if (!value) return undefined
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return undefined
  const year = Number(value.slice(0, 4))
  if (Number.isNaN(year) || year < MIN_YEAR || year > MAX_YEAR) return undefined
  return value
}

export const useSearchFilters = () => {
  const [params, setParams] = useSearchParams()

  const request: SearchRequest = useMemo(() => {
    const pageParam = Number(params.get('page') ?? '1')
    return {
      query: params.get('q') ?? undefined,
      agencies: parseList(params.get('agencies')),
      rfiRfcLabels: parseList(params.get('rfi')),
      topics: [],
      closingAfter: parseDateParam(params.get('closingAfter')),
      closingBefore: parseDateParam(params.get('closingBefore')),
      activeOnly: params.get('active') === 'true',
      page: Number.isNaN(pageParam) || pageParam < 1 ? 1 : pageParam,
      pageSize: 10,
    }
  }, [params])

  const setFilterState = (updates: Partial<SearchRequest>) => {
    const next = new URLSearchParams(params)

    if (updates.query !== undefined) {
      if (updates.query) {
        next.set('q', updates.query)
      } else {
        next.delete('q')
      }
    }

    if (Object.prototype.hasOwnProperty.call(updates, 'agencies')) {
      const serialized = serializeList(updates.agencies)
      if (serialized) {
        next.set('agencies', serialized)
      } else {
        next.delete('agencies')
      }
    }

    if (Object.prototype.hasOwnProperty.call(updates, 'rfiRfcLabels')) {
      const serialized = serializeList(updates.rfiRfcLabels)
      if (serialized) {
        next.set('rfi', serialized)
      } else {
        next.delete('rfi')
      }
    }

    if (Object.prototype.hasOwnProperty.call(updates, 'closingAfter')) {
      const value = updates.closingAfter && parseDateParam(updates.closingAfter)
      if (value) {
        next.set('closingAfter', value)
      } else {
        next.delete('closingAfter')
      }
    }

    if (Object.prototype.hasOwnProperty.call(updates, 'closingBefore')) {
      const value = updates.closingBefore && parseDateParam(updates.closingBefore)
      if (value) {
        next.set('closingBefore', value)
      } else {
        next.delete('closingBefore')
      }
    }

    if (Object.prototype.hasOwnProperty.call(updates, 'activeOnly')) {
      const value = updates.activeOnly
      if (value) {
        next.set('active', 'true')
      } else {
        next.delete('active')
      }
    }

    if (Object.prototype.hasOwnProperty.call(updates, 'page')) {
      const pageValue = updates.page ?? 1
      const page = Math.max(1, pageValue)
      if (page === 1) {
        next.delete('page')
      } else {
        next.set('page', String(page))
      }
    }

    setParams(next, { replace: true })
  }

  const resetFilters = () => {
    const next = new URLSearchParams()
    if (request.query) next.set('q', request.query)
    const serializedRfiLabels = serializeList(request.rfiRfcLabels)
    if (serializedRfiLabels) next.set('rfi', serializedRfiLabels)
    setParams(next, { replace: true })
  }

  return {
    dateBounds: {
      min: `${MIN_YEAR}-01-01`,
      max: `${MAX_YEAR}-12-31`,
    },
    request,
    setFilterState,
    resetFilters,
  }
}
