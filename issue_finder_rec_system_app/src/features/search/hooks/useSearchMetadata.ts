import { useQuery } from '@tanstack/react-query'
import { fetchSearchMetadata } from '../../../services/rules.service'
import type { SearchMetadata } from '../../../lib/types'

export const useSearchMetadata = () =>
  useQuery<SearchMetadata>({
    queryKey: ['search-metadata'],
    queryFn: fetchSearchMetadata,
    staleTime: 1000 * 60 * 60,
  })
