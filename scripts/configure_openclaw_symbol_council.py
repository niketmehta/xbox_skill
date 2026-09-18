"""Restrict inbound OpenClaw WhatsApp DMs to the app's approved target."""

import json
import re
import subprocess
from pathlib import Path


E164_RE = re.compile(r"^\+[1-9][0-9]{7,14}$")


def read_env(path: Path) -> dict:
    values = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def main() -> int:
    env = read_env(Path(__file__).resolve().parents[1] / ".env")
    configured = env.get("OPENCLAW_ALLOWED_TARGETS") or env.get("OPENCLAW_WHATSAPP_TARGET", "")
    targets = [item.strip() for item in configured.split(",") if item.strip()]
    if not targets or any(not E164_RE.fullmatch(item) for item in targets):
        raise SystemExit("OPENCLAW_ALLOWED_TARGETS must contain only E.164 phone numbers")

    operations = {
        "channels.whatsapp.dmPolicy": "allowlist",
        "channels.whatsapp.allowFrom": targets,
        "channels.whatsapp.accounts.default.dmPolicy": "allowlist",
        "channels.whatsapp.accounts.default.allowFrom": targets,
        "channels.whatsapp.selfChatMode": True,
        "channels.whatsapp.accounts.default.selfChatMode": True,
    }
    for path, value in operations.items():
        subprocess.run(
            ["openclaw", "config", "set", path, json.dumps(value), "--strict-json"],
            check=True,
        )

    redacted = [f"***{target[-4:]}" for target in targets]
    print(f"Inbound WhatsApp allowlist configured for {', '.join(redacted)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
