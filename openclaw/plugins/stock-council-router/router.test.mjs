import assert from "node:assert/strict";
import test from "node:test";

import { handleStockCouncilDispatch } from "./router.js";

test("returns the council preview for an inbound WhatsApp ticker", async () => {
  let requestedUrl;
  const response = await handleStockCouncilDispatch(
    { channel: "whatsapp", content: "NVDA", isGroup: false },
    {},
    {
      fetchImpl: async (url) => {
        requestedUrl = url;
        return {
          ok: true,
          json: async () => ({ message_preview: "COUNCIL HOLD: NVDA (WEEK)" }),
        };
      },
    },
  );

  assert.equal(requestedUrl.pathname, "/api/recommendations/symbol/NVDA");
  assert.equal(requestedUrl.searchParams.get("horizon"), "WEEK");
  assert.deepEqual(response, {
    handled: true,
    text: "COUNCIL HOLD: NVDA (WEEK)",
  });
});

test("ignores messages that are not direct WhatsApp ticker requests", async () => {
  assert.equal(
    await handleStockCouncilDispatch({ channel: "telegram", content: "NVDA" }),
    undefined,
  );
  assert.equal(
    await handleStockCouncilDispatch({
      channel: "whatsapp",
      content: "NVDA",
      isGroup: true,
    }),
    undefined,
  );
});
