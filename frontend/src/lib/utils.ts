import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

/** Merge conditional Tailwind classes into a single deduplicated string. */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}
