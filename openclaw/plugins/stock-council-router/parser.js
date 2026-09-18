const SYMBOL_RE = /^[A-Z][A-Z0-9.-]{0,9}$/;
const NON_SYMBOL_WORDS = new Set([
  "A", "AN", "AND", "BUY", "COUNCIL", "HELLO", "HELP", "HI", "HOLD",
  "MONTH", "MONTHLY", "NO", "PLEASE", "RECOMMEND", "RECOMMENDATION", "SELL",
  "THANKS", "THE", "WEEK", "WEEKLY", "YES",
]);

export function parseTickerRequest(content) {
  const text = String(content || "").trim();
  if (!text) return null;

  const horizon = /\b(month|monthly)\b/i.test(text) ? "MONTH" : "WEEK";
  const exact = text.match(/^\$?([A-Za-z][A-Za-z0-9.-]{0,9})(?:\s+(?:week|weekly|month|monthly))?[?!.]?$/i);
  if (exact) {
    const symbol = exact[1].toUpperCase();
    if (SYMBOL_RE.test(symbol) && !NON_SYMBOL_WORDS.has(symbol)) {
      return { symbol, horizon };
    }
  }

  const request = text.match(
    /\b(?:analy[sz]e|buy|sell|hold|council|recommend|recommendation)\s+(?:on\s+)?\$?([A-Za-z][A-Za-z0-9.-]{0,9})\b/i,
  );
  if (!request) return null;
  const symbol = request[1].toUpperCase();
  return SYMBOL_RE.test(symbol) && !NON_SYMBOL_WORDS.has(symbol)
    ? { symbol, horizon }
    : null;
}
