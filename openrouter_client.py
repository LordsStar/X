from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/auth/key"


class OpenRouterError(RuntimeError):
    pass


def verify_api_key(api_key: str, timeout: int = 20) -> dict[str, Any]:
    if not api_key or api_key in {"None", "null"}:
        raise OpenRouterError("No se encontró una API key de OpenRouter.")
    response = requests.get(
        OPENROUTER_KEY_URL,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        timeout=timeout,
    )
    if not response.ok:
        raise OpenRouterError(
            f"OpenRouter rechazó la API key (HTTP {response.status_code})."
        )
    try:
        data = response.json().get("data", response.json())
    except ValueError as exc:
        raise OpenRouterError("OpenRouter no devolvió una verificación válida.") from exc
    return {
        "valid": True,
        "label": data.get("label"),
        "limit": data.get("limit"),
        "usage": data.get("usage"),
        "is_free_tier": data.get("is_free_tier"),
    }


def list_models(api_key: str = "", timeout: int = 30) -> list[dict[str, Any]]:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = requests.get(OPENROUTER_MODELS_URL, headers=headers, timeout=timeout)
    if not response.ok:
        raise OpenRouterError(f"No se pudo cargar el catálogo de modelos ({response.status_code}).")
    try:
        models = response.json()["data"]
    except (ValueError, KeyError, TypeError) as exc:
        raise OpenRouterError("Catálogo de modelos inválido.") from exc
    output = []
    for item in models:
        model_id = item.get("id")
        if model_id:
            output.append(
                {
                    "id": model_id,
                    "name": item.get("name") or model_id,
                    "context_length": item.get("context_length"),
                    "pricing": item.get("pricing", {}),
                }
            )
    return sorted(output, key=lambda item: item["id"].lower())


def _message_text(message: dict[str, Any]) -> str:
    """Normaliza contenido OpenRouter, que puede llegar como texto o bloques."""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                value = block.get("text") or block.get("content")
                if isinstance(value, str):
                    parts.append(value)
        return "\n".join(parts).strip()
    return ""


def _extract_json(text: Any) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise OpenRouterError(
            "El modelo terminó sin devolver contenido JSON. Intenta nuevamente o cambia de proveedor/modelo."
        )
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise OpenRouterError("OpenRouter no devolvió un objeto JSON.")
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise OpenRouterError(f"JSON inválido devuelto por OpenRouter: {exc}") from exc


def validate_candidate(
    candidate: dict[str, Any],
    api_key: str,
    model: str,
    use_web: bool = True,
    timeout: int = 90,
    profile_name: str = "Estricto general",
    profile_instructions: str = "",
    corrections_context: str = "",
) -> dict[str, Any]:
    if not api_key:
        raise OpenRouterError("Falta OPENROUTER_API_KEY.")
    required = {
        "probability",
        "draw_probability",
        "lineup_or_roster_confirmed",
        "source_urls",
        "confidence",
        "summary",
        "warnings",
    }
    prompt = f"""
Evalúa este candidato deportivo prepartido usando información vigente. No inventes
datos, lesiones, alineaciones ni fuentes. Stake es únicamente la fuente de la cuota,
no es un segundo modelo. Devuelve SOLO JSON válido.

Fecha actual: {datetime.utcnow().isoformat()}Z
Candidato: {json.dumps(candidate, ensure_ascii=False)}
Perfil activo: {profile_name}
Instrucciones especializadas: {profile_instructions}
Correcciones/resultados previos aportados por el usuario (úsalos solo como contexto,
no como garantía de repetición): {corrections_context or 'Sin historial disponible'}

Reglas:
- probability: probabilidad decimal de victoria de la selección (0 a 1). Para DNB
  no incluyas el empate dentro de probability; repórtalo por separado.
- draw_probability: probabilidad decimal de empate si aplica; null en deportes sin empate.
- lineup_or_roster_confirmed: true únicamente si existe confirmación actual verificable.
- source_urls: lista de 1 a 5 URLs directas y verificables; no incluyas buscadores.
- confidence: número de 0 a 1 que representa calidad de la evidencia.
- summary: máximo 500 caracteres, en español.
- warnings: lista breve de riesgos concretos.
- Si no puedes verificar información actual, usa confidence <= 0.45,
  lineup_or_roster_confirmed=false y explica la limitación.

Esquema exacto:
{{"probability":0.0,"draw_probability":null,
"lineup_or_roster_confirmed":false,"source_urls":[],"confidence":0.0,
"summary":"","warnings":[]}}
""".strip()
    payload: dict[str, Any] = {
        "model": model,
        "temperature": 0.1,
        "messages": [
            {
                "role": "system",
                "content": "Eres un analista cuantitativo prudente. Respondes solo JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
    }
    # Los modelos Sonar ya incluyen búsqueda; añadir el plugin web puede duplicarla
    # o ser rechazado por algunos proveedores.
    if use_web and not model.startswith("perplexity/"):
        payload["plugins"] = [{"id": "web", "max_results": 5}]
    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://streamlit.io",
            "X-Title": "Stake Direct v2 Strict",
        },
        json=payload,
        timeout=timeout,
    )
    if not response.ok and response.status_code == 400:
        # Algunos modelos no admiten response_format aunque obedecen el esquema del prompt.
        fallback_payload = dict(payload)
        fallback_payload.pop("response_format", None)
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://streamlit.io",
                "X-Title": "Stake Direct v2 Strict",
            },
            json=fallback_payload,
            timeout=timeout,
        )
    if not response.ok:
        detail = response.text[:700]
        raise OpenRouterError(f"OpenRouter respondió HTTP {response.status_code}: {detail}")
    try:
        body = response.json()
        message = body["choices"][0]["message"]
        content = _message_text(message)
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise OpenRouterError("Respuesta inesperada de OpenRouter.") from exc
    if not content:
        # Algunos proveedores consumen la respuesta al usar búsqueda web o salida
        # estructurada. Reintentamos una vez sin esas extensiones.
        retry_payload = dict(payload)
        retry_payload.pop("plugins", None)
        retry_payload.pop("response_format", None)
        retry_response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://streamlit.io",
                "X-Title": "Stake Direct v2 Strict",
            },
            json=retry_payload,
            timeout=timeout,
        )
        if not retry_response.ok:
            raise OpenRouterError(
                f"OpenRouter no pudo reintentar la respuesta vacía "
                f"(HTTP {retry_response.status_code}): {retry_response.text[:500]}"
            )
        try:
            retry_body = retry_response.json()
            content = _message_text(retry_body["choices"][0]["message"])
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise OpenRouterError("Respuesta inesperada de OpenRouter en el reintento.") from exc
    result = _extract_json(content)
    missing = required - result.keys()
    if missing:
        raise OpenRouterError(f"Faltan campos en la validación: {sorted(missing)}")
    result["probability"] = float(result["probability"])
    result["confidence"] = float(result["confidence"])
    if result["draw_probability"] is not None:
        result["draw_probability"] = float(result["draw_probability"])
    for key in ("probability", "confidence"):
        if not 0 <= result[key] <= 1:
            raise OpenRouterError(f"{key} quedó fuera de 0–1.")
    if result["draw_probability"] is not None and not 0 <= result["draw_probability"] <= 1:
        raise OpenRouterError("draw_probability quedó fuera de 0–1.")
    if not isinstance(result["source_urls"], list):
        raise OpenRouterError("source_urls debe ser una lista.")
    if (
        result["draw_probability"] is not None
        and result["probability"] + result["draw_probability"] > 1.0001
    ):
        raise OpenRouterError("Victoria + empate supera 100%.")
    result["source"] = f"OpenRouter:{model}"
    result["model_id"] = model
    return result


def build_consensus(
    analyses: list[dict[str, Any]], max_spread: float
) -> dict[str, Any]:
    if len(analyses) < 2:
        raise OpenRouterError("El consenso requiere por lo menos dos modelos válidos.")
    probabilities = sorted(float(item["probability"]) for item in analyses)
    midpoint = len(probabilities) // 2
    probability = (
        probabilities[midpoint]
        if len(probabilities) % 2
        else (probabilities[midpoint - 1] + probabilities[midpoint]) / 2
    )
    draws = [
        float(item["draw_probability"])
        for item in analyses
        if item.get("draw_probability") is not None
    ]
    draw_probability = sum(draws) / len(draws) if draws else None
    spread = max(probabilities) - min(probabilities)
    urls = []
    for item in analyses:
        for url in item.get("source_urls", []):
            if url not in urls:
                urls.append(url)
    return {
        "probability": probability,
        "draw_probability": draw_probability,
        "lineup_or_roster_confirmed": all(
            bool(item.get("lineup_or_roster_confirmed")) for item in analyses
        ),
        "source_urls": urls[:12],
        "confidence": sum(float(item.get("confidence", 0)) for item in analyses)
        / len(analyses),
        "summary": " | ".join(str(item.get("summary", "")) for item in analyses),
        "warnings": [
            warning
            for item in analyses
            for warning in item.get("warnings", [])
        ],
        "source": "Consenso: " + ", ".join(item.get("model_id", "modelo") for item in analyses),
        "model_spread": spread,
        "consensus_valid": spread <= max_spread,
        "individual_analyses": analyses,
    }


def judge_consensus(
    candidate: dict[str, Any],
    consensus: dict[str, Any],
    api_key: str,
    model: str,
    profile_name: str,
    timeout: int = 90,
) -> dict[str, Any]:
    prompt = f"""
Actúa como juez independiente de una apuesta deportiva. Revisa el candidato y el
consenso de otros modelos. No recalcules ni inventes datos. Rechaza si las fuentes no
respaldan la conclusión, si hay contradicciones, si no están confirmados los
participantes o si la incertidumbre es material. Perfil: {profile_name}.

Candidato: {json.dumps(candidate, ensure_ascii=False)}
Consenso: {json.dumps(consensus, ensure_ascii=False)}

Devuelve SOLO JSON:
{{"approved":false,"confidence":0.0,"reason":"","warnings":[]}}
""".strip()
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "Eres un auditor de riesgo conservador. Solo JSON."},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
    }
    response = requests.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    if not response.ok:
        raise OpenRouterError(f"El juez respondió HTTP {response.status_code}: {response.text[:500]}")
    try:
        message = response.json()["choices"][0]["message"]
        result = _extract_json(_message_text(message))
        return {
            "approved": bool(result["approved"]),
            "confidence": float(result["confidence"]),
            "reason": str(result["reason"]),
            "warnings": list(result.get("warnings", [])),
            "model": model,
        }
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise OpenRouterError("Respuesta inválida del juez final.") from exc
