import { useQuery } from '@tanstack/react-query'
import { fetchAllUserProfiles, fetchRecommendations, fetchUserProfile } from '../../../services/recommendations.service'
import type { Recommendation, UserProfile } from '../../../lib/types'

export const useRecommendationProfiles = () =>
  useQuery<UserProfile[]>({
    queryKey: ['recommendation-profiles'],
    queryFn: fetchAllUserProfiles,
  })

export const useRecommendations = (userId?: string) => {
  const profileQuery = useQuery({
    queryKey: ['user-profile', userId],
    queryFn: () => fetchUserProfile(userId!),
    enabled: Boolean(userId),
  })

  const recommendationsQuery = useQuery({
    queryKey: ['recommendations', userId],
    queryFn: () => fetchRecommendations(userId!),
    enabled: Boolean(userId),
    select: (items) => items.slice().sort((a, b) => b.score - a.score) as Recommendation[],
  })

  return {
    user: profileQuery.data,
    isUserLoading: profileQuery.isPending,
    recommendations: recommendationsQuery.data,
    isRecommendationsLoading: recommendationsQuery.isPending,
    isFetching: profileQuery.isFetching || recommendationsQuery.isFetching,
    error: profileQuery.error ?? recommendationsQuery.error,
  }
}
