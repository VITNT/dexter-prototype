export function hasSufficientTopTwoMargin(rankedPredictions, minimumMargin = 0) {
  if (!Array.isArray(rankedPredictions) || rankedPredictions.length === 0) return false;
  const best = Number(rankedPredictions[0]?.conf ?? rankedPredictions[0]?.confidence);
  const second = Number(rankedPredictions[1]?.conf ?? rankedPredictions[1]?.confidence ?? 0);
  if (!Number.isFinite(best) || !Number.isFinite(second)) return false;
  return best - second >= Math.max(0, Number(minimumMargin) || 0);
}
