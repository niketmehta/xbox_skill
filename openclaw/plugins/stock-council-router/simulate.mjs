import { handleStockCouncilDispatch } from "./router.js";

const content = process.argv.slice(2).join(" ").trim() || "NVDA";
const response = await handleStockCouncilDispatch(
  { channel: "whatsapp", content, isGroup: false },
  { channelId: "whatsapp" },
);

if (!response?.handled || !response.text) {
  console.error("Simulation did not produce a handled WhatsApp response.");
  process.exit(1);
}

console.log(response.text);
