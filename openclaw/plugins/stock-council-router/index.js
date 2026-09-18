import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import { parseTickerRequest } from "./parser.js";

const COUNCIL_BASE_URL = "http://127.0.0.1:5001";

export default definePluginEntry({
  id: "stock-council-router",
  name: "Stock Council Router",
  description: "Replies to WhatsApp ticker requests without invoking a language model.",
  register(api) {
    api.on("before_dispatch", async (event, ctx) => {
      const channel = String(event.channel || ctx.channelId || "").toLowerCase();
      if (channel !== "whatsapp" || event.isGroup) return;

      const request = parseTickerRequest(event.content);
      if (!request) return;

      const url = new URL(
        `/api/recommendations/symbol/${encodeURIComponent(request.symbol)}`,
        COUNCIL_BASE_URL,
      );
      url.searchParams.set("horizon", request.horizon);

      try {
        const response = await fetch(url, { signal: AbortSignal.timeout(180_000) });
        const result = await response.json();
        if (!response.ok || !result.message_preview) {
          return {
            handled: true,
            text: `Council could not analyze ${request.symbol}: ${result.error || response.statusText}.`,
          };
        }
        return { handled: true, text: String(result.message_preview) };
      } catch (error) {
        api.logger.error(`Stock council request failed for ${request.symbol}: ${error}`);
        return {
          handled: true,
          text: `Council service is temporarily unavailable for ${request.symbol}. Please try again.`,
        };
      }
    });
  },
});
