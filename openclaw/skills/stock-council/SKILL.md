---
name: stock-council
description: Run the deployed trading council whenever the user sends a stock ticker or asks whether to buy, sell, or hold a specific symbol.
user-invocable: true
---

# Stock council

Use this skill for a direct stock-symbol request received through WhatsApp, including
plain messages such as `NVDA`, `$AAPL`, `MSFT week`, or `Should I buy AMD?`.

1. Extract exactly one ticker and uppercase it. Remove one leading `$`. Accept only
   `A-Z`, digits, dot, and hyphen, with a maximum length of 10 characters.
2. Use `WEEK` unless the user explicitly says `month` or `monthly`; then use `MONTH`.
3. If there is no clear ticker or there are multiple tickers, ask for exactly one
   ticker and do not run a command.
4. Run this command with the validated values:

   `/home/trader/xbox_skill/.venv/bin/python /home/trader/xbox_skill/scripts/whatsapp_symbol_council.py SYMBOL --horizon HORIZON`

5. Return the command's stdout as the reply. Do not reinterpret or override its
   verdict. If it fails, return its error text.

This workflow is analysis-only. Never place, submit, modify, or cancel an order in
response to a symbol message.
