from __future__ import annotations

from typing import Any


def evaluate(
    candidate: dict[str, Any],
    model: dict[str, Any] | None,
    config: dict[str, Any],
    bankroll: float,
    pending_total: float,
    sport_pending: float = 0.0,
    profile: dict[str, Any] | None = None,
    judge: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = dict(candidate)
    reasons: list[str] = []
    warnings: list[str] = []
    if row["market_overround"] > config["market_quality"]["max_overround"]:
        reasons.append("Margen del mercado superior al permitido")
    if bankroll > 0 and pending_total / bankroll > config["risk"]["max_total_exposure"]:
        reasons.append("Exposición total superior al límite")
    if bankroll > 0 and sport_pending / bankroll > config["risk"]["max_sport_exposure"]:
        reasons.append("Exposición del deporte superior al límite")
    if not model:
        row.update(status="Pendiente", reasons=reasons or ["Falta validación externa"])
        return row

    probability = float(model["probability"])
    draw_probability = model.get("draw_probability")
    is_dnb = row["market"].strip().lower() == "draw no bet"
    if is_dnb and draw_probability is not None and draw_probability < 1:
        # Stake devuelve la apuesta en el empate. Su mercado de-vig representa la
        # probabilidad condicional P(victoria | no empate), no la P(victoria) cruda.
        comparable_probability = probability / (1 - draw_probability)
        ev = probability * row["odds"] + draw_probability - 1
    else:
        comparable_probability = probability
        ev = probability * row["odds"] - 1
    divergence = abs(comparable_probability - row["market_no_vig_probability"])
    sources = [url for url in model.get("source_urls", []) if str(url).startswith("http")]
    row.update(
        second_model_probability=round(probability, 4),
        comparable_model_probability=round(comparable_probability, 4),
        draw_probability=draw_probability,
        lineup_or_roster_confirmed=bool(model.get("lineup_or_roster_confirmed")),
        second_model_source=model.get("source", "OpenRouter"),
        source_urls=sources,
        model_confidence=round(float(model.get("confidence", 0)), 4),
        analysis_summary=model.get("summary", ""),
        model_warnings=model.get("warnings", []),
        divergence=round(divergence, 4),
        ev=round(ev, 4),
        max_stake=round(bankroll * config["risk"]["max_single_stake"], 2),
    )
    if not sources:
        reasons.append("Sin fuentes externas verificables")
    profile = profile or {}
    min_confidence = float(profile.get("min_confidence", 0.60))
    min_ev = float(profile.get("min_ev", config["validation"]["min_ev"]))
    max_divergence = float(
        profile.get("max_divergence", config["validation"]["max_divergence"])
    )
    if row["model_confidence"] < min_confidence:
        reasons.append("Confianza insuficiente del segundo modelo")
    if not row["lineup_or_roster_confirmed"]:
        reasons.append("Alineación, roster o participantes sin confirmar")
    if ev < min_ev:
        reasons.append(f"EV inferior al mínimo del perfil ({min_ev:.0%})")
    if divergence > max_divergence:
        reasons.append(f"Divergencia superior al límite ({max_divergence:.0%})")
    if model.get("consensus_valid") is False:
        reasons.append("Los modelos discrepan más de lo permitido")
    if judge is not None:
        row["judge"] = judge
        if not judge.get("approved"):
            reasons.append(f"Juez final rechazó: {judge.get('reason', 'sin detalle')}")
        if float(judge.get("confidence", 0)) < min_confidence:
            reasons.append("Confianza insuficiente del juez final")
    if row["sport"] == "soccer" and not row["protected_market"]:
        if draw_probability is None:
            reasons.append("Falta probabilidad de empate")
        elif draw_probability >= config["soccer"]["draw_probability_dnb_required"]:
            reasons.append("Riesgo de empate exige DNB o doble oportunidad")
        elif draw_probability >= config["soccer"]["draw_probability_confidence_reduction_from"]:
            warnings.append("Confianza reducida por riesgo de empate")
    row["warnings"] = warnings
    row["reasons"] = reasons
    row["status"] = "Aprobado" if not reasons else "Descartado"
    return row


def rank_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            row.get("status") != "Aprobado",
            -row.get("ev", -99),
            -row.get("market_no_vig_probability", 0),
            row.get("start_rd", ""),
        ),
    )
