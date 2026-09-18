import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import { handleStockCouncilDispatch } from "./router.js";

export default definePluginEntry({
  id: "stock-council-router",
  name: "Stock Council Router",
  description: "Replies to WhatsApp ticker requests without invoking a language model.",
  register(api) {
    api.on("before_dispatch", (event, ctx) =>
      handleStockCouncilDispatch(event, ctx, { logger: api.logger }),
    );
  },
});
