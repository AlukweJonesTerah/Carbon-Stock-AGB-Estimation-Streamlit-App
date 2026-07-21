import os
from google import genai


# Set GEMINI_API_KEY outside source control, for example in PowerShell:
# $env:GEMINI_API_KEY = "your-key"
api_key = os.getenv("GEMINI_API_KEY", "").strip()

if not api_key or api_key == "GEMINI_API_KEY":
    raise RuntimeError(
        "Set a valid GEMINI_API_KEY environment variable before running this script."
    )

client = genai.Client(api_key=api_key)

response_stream = client.interactions.create(
    model="gemini-3.6-flash",
    input="Analyze our 2026 Q2 financial sheets, find anomalies, and draft an email.",
    system_instruction="You are an autonomous financial auditor agent.",
    store=False,
    stream=True,
)

for event in response_stream:
    if event.event_type == "step.delta" and event.delta.type == "text":
        print(event.delta.text, end="", flush=True)
    elif event.event_type == "interaction.completed":
        interaction = event.interaction
        usage = getattr(interaction, "usage", None)
        total_tokens = getattr(usage, "total_tokens", None)

        if total_tokens is not None:
            print(f"\n\nTask finished. Total tokens: {total_tokens}")
        else:
            print("\n\nTask finished.")
