import assert from "node:assert/strict";
import test from "node:test";

import { parseTickerRequest } from "./parser.js";

test("parses plain and dollar-prefixed symbols", () => {
  assert.deepEqual(parseTickerRequest("NVDA"), { symbol: "NVDA", horizon: "WEEK" });
  assert.deepEqual(parseTickerRequest("$aapl month"), { symbol: "AAPL", horizon: "MONTH" });
  assert.deepEqual(parseTickerRequest("BRK.B weekly"), { symbol: "BRK.B", horizon: "WEEK" });
});

test("parses natural recommendation requests", () => {
  assert.deepEqual(parseTickerRequest("Should I buy AMD?"), { symbol: "AMD", horizon: "WEEK" });
  assert.deepEqual(parseTickerRequest("analyze msft monthly"), { symbol: "MSFT", horizon: "MONTH" });
});

test("ignores ordinary chat and ambiguous requests", () => {
  assert.equal(parseTickerRequest("hello"), null);
  assert.equal(parseTickerRequest("thanks"), null);
  assert.equal(parseTickerRequest("NVDA AAPL"), null);
});
