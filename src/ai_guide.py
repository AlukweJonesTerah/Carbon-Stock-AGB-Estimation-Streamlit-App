import json
import base64
import urllib.request
import urllib.error
import streamlit as st
from src.config import *

def build_learning_context(params):
    """Return only the current, non-sensitive analysis facts needed by the guide."""
    context = {
        "reference_year": params.get("agb_year"),
        "selected_counties": list(params.get("county_selection", [])),
        "sample_pixels": params.get("num_pixels"),
        "training_split": params.get("train_split"),
        "models": ["Random Forest", "Gradient Tree Boosting", "Support Vector Machine", "Smart Weighted Ensemble"],
        "carbon_to_biomass_factor": CARBON_TO_BIOMASS_FACTOR,
        "predictors_used": ["Sentinel-2 Multispectral", "NDVI, EVI, SAVI", "SRTM Elevation, Slope, Aspect", "Sentinel-1 Radar (VV & VH)", "Lang Canopy Height", "ERA5-Land Climate", "OpenLandMap Soil Carbon & Clay", "PALSAR Radar", "MODIS LST & GPP", "Hansen Treecover & Loss", "ESA WorldCover", "CSP Human Modification Index", "RESOLVE Biomes", "Coordinates"],
    }
    validation = st.session_state.get("validation_results")
    if validation:
        context["validation"] = {
            name: {key: result.get(key) for key in ("rmse", "mae", "bias", "mape", "r2")}
            for name, result in validation.items()
        }
    if st.session_state.get("mean_spread") is not None:
        context["mean_model_spread_tC_ha"] = st.session_state["mean_spread"]
    
    reg_stats = st.session_state.get("regional_stats")
    if reg_stats:
        context["regional_statistics"] = reg_stats
        
    restoration = st.session_state.get("restoration_scenario")
    if restoration:
        # Exclude the large growth_curve array to save tokens
        rest_clean = {k: v for k, v in restoration.items() if k != "growth_curve"}
        context["restoration_scenario_results"] = rest_clean
        
    return json.dumps(context, default=str)


def ask_learning_guide(provider, model, user_message, chat_history, context, gemini_key="",
                       language="English", audio_bytes=None, audio_mime_type="audio/wav"):
    """Ask Gemini, with a local Ollama fallback option, without persisting credentials."""
    instructions = """
You are the Environmental & Carbon Learning Guide inside a Kenya carbon-stock and above-ground
biomass app. Teach someone with no technical background using warm, plain language and short
paragraphs. You can explain: climate change, greenhouse gases, carbon emissions and emitters,
carbon footprints, forests and nature-based solutions, carbon stock and AGB, offsets and carbon
credits, voluntary and compliance carbon markets, additionality, leakage, permanence, monitoring,
verification, and responsible ways people, organisations, and governments can reduce emissions.

Explain maps, colours, units, uncertainty, validation metrics, data sources, model outputs, regional statistics, and restoration scenarios (including logistic growth and biome carrying capacity limits).
Use the supplied app context only for current run details. Clearly distinguish estimates from
measurements. Never present an output as a verified carbon credit, market price, investment
recommendation, legal conclusion, or certification decision. For current carbon prices, laws,
policies, named project claims, or recent events, say that they change over time and recommend
checking an authoritative current source. Do not invent statistics or sources. For questions about
major emitters, discuss sectors and drivers without shaming individuals or making unsupported
claims about a company or community. End map interpretations with one practical next step.
For beginner questions, give a complete answer in 4–6 short paragraphs.
Define unfamiliar terms, use a simple analogy, and finish with a practical next step.
Never end mid-sentence. If the topic has several parts, use concise headings or bullets.
""".strip()
    language_instruction = {
        "English": "Reply in clear English.",
        "Kiswahili": "Jibu kwa Kiswahili rahisi na wazi.",
        "English + Kiswahili": "Give each key explanation first in clear English, then in clear Kiswahili.",
    }.get(language, "Reply in clear English.")
    messages = [{"role": "system", "content": instructions + "\n\n" + language_instruction + "\n\nCurrent app context:\n" + context}]
    messages.extend({"role": item["role"], "content": item["content"]} for item in chat_history[-10:])
    messages.append({"role": "user", "content": user_message})

    if provider == "Gemini":
        if not gemini_key:
            raise ValueError("Add a Gemini API key in the guide settings or Streamlit secrets.")
        contents = []
        for message in messages[1:]:
            contents.append({
                "role": "model" if message["role"] == "assistant" else "user",
                "parts": [{"text": message["content"]}],
            })
        if audio_bytes:
            contents[-1]["parts"].append({
                "inline_data": {
                    "mime_type": audio_mime_type,
                    "data": base64.b64encode(audio_bytes).decode("ascii"),
                }
            })
        payload = json.dumps({
            "system_instruction": {"parts": [{"text": messages[0]["content"]}]},
            "contents": contents,
            "generationConfig": {"temperature": 0.35, "maxOutputTokens": 1200},
        }).encode("utf-8")
        request = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            data=payload,
            headers={"Content-Type": "application/json", "x-goog-api-key": gemini_key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini request failed ({error.code}): {detail[:300]}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Could not reach Gemini: {error.reason}") from error
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as error:
            raise RuntimeError("Gemini returned no text response for this question.") from error

    if audio_bytes:
        raise RuntimeError("Voice questions require Gemini. Use a typed question with the local Ollama fallback.")
    payload = json.dumps({"model": model, "messages": messages, "stream": False}).encode("utf-8")
    request = urllib.request.Request(
        "http://localhost:11434/api/chat", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data["message"]["content"]
    except urllib.error.URLError as error:
        raise RuntimeError(
            "Could not reach Ollama at http://localhost:11434. Start Ollama and pull the selected model first."
        ) from error


def offline_learning_response(question):
    """Provide a useful, no-network explanation while AI services are unavailable."""
    query = question.lower()
    if any(word in query for word in ("feature", "used", "predictor", "data source")):
        return (
            "**This app combines several kinds of information to estimate carbon.**\n\n"
            "- **Sentinel-1 radar** helps describe vegetation structure, even through clouds.\n"
            "- **Sentinel-2 imagery** provides colour-based vegetation measures such as NDVI.\n"
            "- **Terrain, rainfall, temperature, soil, canopy height, and land-surface temperature** help explain why biomass differs from place to place.\n"
            "- **Random Forest, Gradient Tree Boosting, and SVM** are three different prediction methods. The ensemble is their average.\n\n"
            "These are estimates, not direct field measurements. A good next step is to open **Variable Importance** to see which inputs mattered most in this run."
        )
    if any(word in query for word in ("map", "colour", "color", "legend")):
        return (
            "**How to read the map:** use the colour bar beside the map. Each colour represents a range of estimated carbon stock or AGB. "
            "Read the number and unit on the legend before deciding whether a colour is high or low. County outlines only show boundaries; they do not change the estimate. "
            "Turn on **confidence classes** to see where the three models agree most."
        )
    if any(word in query for word in ("credit", "market", "offset")):
        return (
            "A **carbon credit** normally represents one tonne of verified carbon dioxide equivalent reduced or removed under a recognised method. "
            "This app estimates carbon stock; it does **not** create, verify, price, or certify credits. A credible project also needs a baseline, additionality, monitoring, verification, and checks for leakage and permanence."
        )
    if any(word in query for word in ("climate", "emission", "emitter", "greenhouse")):
        return (
            "Climate change is driven mainly by greenhouse gases accumulating in the atmosphere. Key sources include energy, transport, industry, agriculture, land-use change, and waste. "
            "The most reliable action is usually to reduce emissions at the source first; protecting or restoring ecosystems can complement, but not replace, those reductions."
        )
    return (
        "The online guide is not connected yet, but I can still help with the app basics. Try asking about **map colours**, **features used**, **validation**, **carbon credits**, or **climate emissions**. "
        "For open-ended questions, add a Gemini key in the guide settings or start Ollama locally."
    )


def render_voice_player(text, language):
    """Offer browser-native text-to-speech without an additional cloud service."""
    language_code = "sw-KE" if language == "Kiswahili" else "en-KE"
    safe_text = json.dumps(text)
    st.iframe(
        f"""
        <style>
          body {{ margin: 0; font-family: sans-serif; background: transparent; }}
          button {{ background: #2d6a4f; color: white; border: 0; border-radius: 8px; padding: 8px 14px; font-weight: 700; cursor: pointer; }}
        </style>
        <button onclick='window.speechSynthesis.cancel(); const speech = new SpeechSynthesisUtterance({safe_text}); speech.lang = "{language_code}"; window.speechSynthesis.speak(speech);'>🔊 Listen to the latest answer</button>
        """,
        height=48,
    )

