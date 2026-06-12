/**
 * ratingStore — per-sample 0–3 star ratings, persisted to localStorage.
 *
 * Keys are canonical sourceKeys (`tiled:<server>:<path>` or `local:<path>`).
 * 0 = unrated; 1–3 = star count.
 */
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export type StarRating = 0 | 1 | 2 | 3;

interface RatingState {
  ratings: Record<string, StarRating>;
  setRating: (sourceKey: string, rating: StarRating) => void;
  getRating: (sourceKey: string) => StarRating;
}

export const useRatingStore = create<RatingState>()(
  persist(
    (set, get) => ({
      ratings: {},

      setRating: (sourceKey, rating) =>
        set((s) => ({ ratings: { ...s.ratings, [sourceKey]: rating } })),

      getRating: (sourceKey) => get().ratings[sourceKey] ?? 0,
    }),
    { name: 'sam3_ratings' },
  ),
);
