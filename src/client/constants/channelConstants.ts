// Channel layout of the automata state.
// The first VISIBLE_CHANNELS are rendered as RGB. A config may add up to
// MAX_HIDDEN_CHANNELS invisible per-cell memory channels that only feed back
// through the kernel. The channel count of a config is implied by the shape
// of its weights array: weights.length == total channels.
export const VISIBLE_CHANNELS = 3;
export const MAX_HIDDEN_CHANNELS = 13;
export const MAX_TOTAL_CHANNELS = VISIBLE_CHANNELS + MAX_HIDDEN_CHANNELS;
export const KERNEL_SIZE = 5;

export function clampChannelCount(count: number): number {
  if (!Number.isFinite(count)) return VISIBLE_CHANNELS;
  return Math.max(VISIBLE_CHANNELS, Math.min(MAX_TOTAL_CHANNELS, Math.floor(count)));
}
