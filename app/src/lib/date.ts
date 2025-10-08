const formatter = new Intl.DateTimeFormat('en-US', {
  month: 'short',
  day: 'numeric',
  year: 'numeric',
})

export const formatDate = (isoDate?: string | null) => {
  if (!isoDate) return '—'
  const date = new Date(isoDate)
  if (Number.isNaN(date.getTime())) return '—'
  return formatter.format(date)
}

export const daysUntil = (isoDate?: string | null) => {
  if (!isoDate) return null
  const now = new Date()
  const future = new Date(isoDate)
  if (Number.isNaN(future.getTime())) return null
  const diff = future.getTime() - now.getTime()
  return Math.ceil(diff / (1000 * 60 * 60 * 24))
}
