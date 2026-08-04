/**
 * StarRating — clickable 1–3 star widget.
 * Clicking the same star that's already set clears the rating back to 0.
 */
import { Star } from '@phosphor-icons/react';
import type { StarRating as StarRatingValue } from '@/stores/ratingStore';

interface StarRatingProps {
  value: StarRatingValue;
  onChange: (rating: StarRatingValue) => void;
  size?: number;
  /** If true, the stars are read-only (no onClick). */
  readonly?: boolean;
}

export default function StarRating({ value, onChange, size = 13, readonly = false }: StarRatingProps) {
  return (
    <div className="flex items-center gap-0.5 shrink-0">
      {([1, 2, 3] as StarRatingValue[]).map((star) => {
        const filled = value >= star;
        return (
          <button
            key={star}
            type="button"
            disabled={readonly}
            onClick={(e) => {
              e.stopPropagation();
              onChange(value === star ? 0 : star);
            }}
            className={`transition-colors leading-none ${readonly ? 'cursor-default' : 'cursor-pointer hover:scale-110'}`}
            title={readonly ? undefined : `Rate ${star} star${star > 1 ? 's' : ''}`}
            aria-label={`${star} star${star > 1 ? 's' : ''}`}
          >
            <Star
              size={size}
              weight={filled ? 'fill' : 'regular'}
              className={filled ? 'text-amber-400' : 'text-slate-600'}
            />
          </button>
        );
      })}
    </div>
  );
}
