import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

def creator_known_binary(project_name: str) -> int:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ANTHROPIC_API_KEY in environment.")

    client = Anthropic(api_key=api_key)

    prompt = f"""
Return ONLY a single character: 1 or 0.

Definition:
- Return 1 if the creator/founding team behind "{project_name}" is definitely publicly known (real-world identities are publicly documented).
- Return 0 if the creator/team is anonymous, only pseudonymous, disputed, or you are not fully certain.

Rules:
- Output must be exactly one character: 1 or 0.
- No other text, no punctuation, no newlines.
"""

    msg = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=5,
        messages=[{"role": "user", "content": prompt}],
    )

    text = msg.content[0].text.strip()
    return 1 if text.startswith("1") else 0

if __name__ == "__main__":
    print(creator_known_binary("Pi"))
