from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from engine import evaluate, rank_candidates
from openrouter_client import OpenRouterError, build_consensus, judge_consensus, list_models, validate_candidate
from stake_client import StakeClient, StakeError

ROOT = Path(__file__).parent
CONFIG = json.loads((ROOT / "stake_logic_v2_strict.json").read_text(encoding="utf-8"))
PROFILES = json.loads((ROOT / "analysis_profiles.json").read_text(encoding="utf-8"))

st.set_page_config(page_title="Stake Multiagente v3", page_icon="🎯", layout="wide")
st.title("🎯 Corridas Stake — OpenRouter multiagente")
st.caption("Stake directo · Perfiles especializados · Consenso de modelos · Juez independiente")


def secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, os.getenv(name, default)))
    except Exception:
        return os.getenv(name, default)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_models(api_key: str) -> list[dict]:
    return list_models(api_key)


def display_table(rows: list[dict]) -> None:
    if not rows:
        st.info("No hay filas para mostrar.")
        return
    frame = pd.DataFrame(rows)
    columns = {
        "start_rd": "Hora RD", "sport": "Deporte", "event": "Evento",
        "market": "Mercado", "selection": "Selección", "odds": "Cuota",
        "market_no_vig_probability": "Stake sin margen",
        "second_model_probability": "Consenso/modelo", "ev": "EV",
        "model_spread": "Dispersión", "status": "Estado",
    }
    present = [key for key in columns if key in frame.columns]
    view = frame[present].rename(columns=columns)
    for name in ("Stake sin margen", "Consenso/modelo", "EV", "Dispersión"):
        if name in view:
            view[name] = view[name].map(lambda value: f"{float(value):.1%}" if pd.notna(value) else "—")
    if "Hora RD" in view:
        view["Hora RD"] = pd.to_datetime(view["Hora RD"]).dt.strftime("%d/%m %I:%M %p")
    st.dataframe(view, use_container_width=True, hide_index=True)


def history_context(history: list[dict]) -> str:
    safe = [
        {key: item.get(key) for key in ("event", "selection", "odds", "result", "note")}
        for item in history[-20:]
    ]
    return json.dumps(safe, ensure_ascii=False) if safe else ""


for key, default in {"scan": None, "evaluations": [], "history": [], "model_catalog": []}.items():
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.header("Configuración")
    api_key_input = st.text_input("OpenRouter API key", type="password")
    api_key = api_key_input or secret("OPENROUTER_API_KEY")
    if st.button("Actualizar catálogo de modelos", use_container_width=True):
        cached_models.clear()
    try:
        st.session_state.model_catalog = cached_models(api_key)
    except Exception as exc:
        st.warning(f"Catálogo no disponible: {exc}")
    model_ids = [item["id"] for item in st.session_state.model_catalog]
    fallback = secret("OPENROUTER_MODEL", "openai/gpt-4.1-mini")
    if fallback not in model_ids:
        model_ids = [fallback] + model_ids

    mode = st.radio("Modo de análisis", ["Consenso multiagente", "Modelo único"])
    if mode == "Consenso multiagente":
        analyst_models = st.multiselect(
            "Modelos analistas (2–3)", model_ids,
            default=model_ids[:2] if len(model_ids) >= 2 else model_ids,
            max_selections=3,
        )
        judge_model = st.selectbox("Modelo juez", model_ids, index=0)
    else:
        analyst_models = [st.selectbox("Modelo analista", model_ids, index=0)]
        judge_model = analyst_models[0]

    profile_name = st.selectbox("Perfil de análisis", list(PROFILES))
    profile = PROFILES[profile_name]
    st.caption(profile["description"])
    use_web = st.checkbox("Búsqueda web de OpenRouter", value=True)
    bankroll = st.number_input("Banca actual (USD)", min_value=0.0, value=151.04, step=1.0)
    pending = st.number_input("Exposición pendiente (USD)", min_value=0.0, value=0.0, step=1.0)
    hours = st.slider("Ventana futura (horas)", 2, 36, 18)
    max_candidates = st.slider("Candidatos a validar", 1, 10, 4)
    calls = max_candidates * len(analyst_models) + (max_candidates if mode == "Consenso multiagente" else 0)
    st.info(f"La corrida usará aproximadamente {calls} llamadas a OpenRouter.")
    st.write(f"EV mínimo: **{profile['min_ev']:.0%}**")
    st.write(f"Stake máximo: **${bankroll * CONFIG['risk']['max_single_stake']:.2f}**")

tab_run, tab_history, tab_rules = st.tabs(["Corrida", "Resultados y correcciones", "Reglas"])

with tab_history:
    st.subheader("Memoria de resultados")
    st.caption("El historial sirve como contexto. Descárgalo para conservarlo entre reinicios de Streamlit Cloud.")
    uploaded = st.file_uploader("Importar historial JSON", type=["json"])
    if uploaded is not None:
        try:
            payload = json.load(uploaded)
            imported = payload.get("history", payload) if isinstance(payload, dict) else payload
            if not isinstance(imported, list):
                raise ValueError("El JSON debe contener una lista.")
            st.session_state.history = imported
            st.success(f"Se importaron {len(imported)} registros.")
        except Exception as exc:
            st.error(f"No se pudo importar: {exc}")
    with st.form("result_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        event = c1.text_input("Evento")
        selection = c2.text_input("Selección")
        odds = c1.number_input("Cuota", min_value=1.0, value=1.50, step=0.01)
        result = c2.selectbox("Resultado", ["Ganó", "Perdió", "Void", "Pendiente"])
        note = st.text_area("Corrección o nota")
        if st.form_submit_button("Agregar resultado"):
            st.session_state.history.append({
                "created_at": datetime.utcnow().isoformat() + "Z", "event": event,
                "selection": selection, "odds": odds, "result": result, "note": note,
            })
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)
        st.download_button(
            "Descargar historial JSON",
            json.dumps({"history": st.session_state.history}, ensure_ascii=False, indent=2),
            "historial_stake.json", "application/json",
        )

with tab_rules:
    st.json({"logic": CONFIG, "active_profile": profile, "profiles": PROFILES})

with tab_run:
    col_a, col_b = st.columns(2)
    run_scan = col_a.button("🔄 Consultar Stake", type="primary", use_container_width=True)
    validate = col_b.button("🧠 Ejecutar análisis", use_container_width=True, disabled=st.session_state.scan is None)

    if run_scan:
        progress_bar = st.progress(0.0, text="Iniciando consulta directa…")

        def progress(label: str, fraction: float) -> None:
            progress_bar.progress(fraction, text=label)

        try:
            scan = StakeClient().scan(CONFIG, hours_ahead=hours, progress=progress)
            scan["candidates"] = sorted(
                [row for row in scan["candidates"] if row["market_overround"] <= CONFIG["market_quality"]["max_overround"]],
                key=lambda row: (-row["market_no_vig_probability"], row["start_rd"]),
            )
            st.session_state.scan = scan
            st.session_state.evaluations = []
            progress_bar.empty()
            st.success("Consulta directa terminada.")
        except StakeError as exc:
            progress_bar.empty()
            st.error(str(exc))

    scan = st.session_state.scan
    if scan:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Deportes", scan["sports"])
        m2.metric("Eventos próximos", scan["fixtures_remaining"])
        m3.metric("Candidatos", len(scan["candidates"]))
        m4.metric("Fallos", scan["request_failures"])
        st.caption(f"Generado: {scan['generated_rd']} · {scan['source']}")
        display_table(scan["candidates"][:100])

    if validate and scan:
        if not api_key:
            st.error("Configura OPENROUTER_API_KEY.")
        elif mode == "Consenso multiagente" and len(analyst_models) < 2:
            st.error("Selecciona por lo menos dos modelos analistas.")
        else:
            targets = scan["candidates"][:max_candidates]
            results = []
            total_steps = len(targets) * (len(analyst_models) + (1 if mode == "Consenso multiagente" else 0))
            completed = 0
            bar = st.progress(0.0, text="Preparando análisis…")
            context = history_context(st.session_state.history)
            for candidate in targets:
                analyses, errors = [], []
                for analyst_model in analyst_models:
                    bar.progress(completed / max(total_steps, 1), text=f"{analyst_model}: {candidate['selection']}")
                    try:
                        analyses.append(validate_candidate(
                            candidate, api_key, analyst_model, use_web=use_web,
                            profile_name=profile_name,
                            profile_instructions=profile["instructions"],
                            corrections_context=context,
                        ))
                    except (OpenRouterError, ValueError) as exc:
                        errors.append(f"{analyst_model}: {exc}")
                    completed += 1

                judge = None
                model_result = None
                try:
                    if mode == "Consenso multiagente":
                        model_result = build_consensus(analyses, profile["max_model_spread"])
                        bar.progress(completed / max(total_steps, 1), text=f"Juez: {candidate['selection']}")
                        judge = judge_consensus(candidate, model_result, api_key, judge_model, profile_name)
                        completed += 1
                    elif analyses:
                        model_result = analyses[0]
                    result_row = evaluate(candidate, model_result, CONFIG, bankroll, pending, profile=profile, judge=judge)
                except (OpenRouterError, ValueError) as exc:
                    result_row = evaluate(candidate, None, CONFIG, bankroll, pending, profile=profile)
                    errors.append(str(exc))
                if errors:
                    result_row.setdefault("reasons", []).extend(errors)
                    if result_row.get("status") == "Aprobado":
                        result_row["status"] = "Descartado"
                if model_result:
                    result_row["model_spread"] = model_result.get("model_spread")
                    result_row["individual_analyses"] = model_result.get("individual_analyses", analyses)
                results.append(result_row)
            st.session_state.evaluations = rank_candidates(results)
            bar.empty()

    evaluations = st.session_state.evaluations
    if evaluations:
        st.subheader("Resultado")
        approved = [row for row in evaluations if row["status"] == "Aprobado"]
        if approved:
            winner = approved[0]
            st.success(f"PICK APROBADO: {winner['selection']} · {winner['market']} @{winner['odds']:.2f} · máximo ${winner['max_stake']:.2f}")
        else:
            st.warning("SIN PICK: ningún candidato superó modelos, juez y gates matemáticos.")
        display_table(evaluations)
        for row in evaluations:
            with st.expander(f"{row['status']} · {row['selection']} @{row['odds']:.2f}"):
                st.write(row.get("analysis_summary", "Sin resumen"))
                if row.get("judge"):
                    st.write("**Juez:**", row["judge"].get("reason"))
                if row.get("reasons"):
                    st.write("**Motivos:**", "; ".join(row["reasons"]))
                for analysis in row.get("individual_analyses", []):
                    st.markdown(f"**{analysis.get('model_id', 'Modelo')} — {analysis.get('probability', 0):.1%}**")
                    st.write(analysis.get("summary", ""))
                for url in row.get("source_urls", []):
                    st.markdown(f"- [{url}]({url})")
        export = {
            "logic": CONFIG["name"], "version": CONFIG["version"], "profile": profile_name,
            "analyst_models": analyst_models, "judge_model": judge_model,
            "scan": scan, "evaluations": evaluations, "history": st.session_state.history,
        }
        st.download_button(
            "⬇️ Descargar corrida completa",
            json.dumps(export, ensure_ascii=False, indent=2),
            "corrida_stake_multiagente.json", "application/json",
        )

st.divider()
st.caption("Analiza, no coloca apuestas. Las probabilidades no garantizan resultados y SIN PICK es una salida válida.")
