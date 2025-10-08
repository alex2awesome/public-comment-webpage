import { mockApi } from './mockApi'

export const fetchRecommendations = (userId: string) => mockApi.getRecommendations(userId)
export const fetchUserProfile = (userId: string) => mockApi.getUserProfile(userId)
export const fetchAllUserProfiles = () => mockApi.listUserProfiles()
