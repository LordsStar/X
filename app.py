from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from engine import block_opposing_approvals, evaluate, rank_candidates
from openrouter_client import OpenRouterError, build_consensus, judge_consensus, list_models, validate_candidate, verify_api_key
from stake_client import StakeClient, StakeError

ROOT = Path(__file__).parent
APP_VERSION = "3.2.0"
CONFIG = json.loads((ROOT / "stake_logic_v2_strict.json").read_text(encoding="utf-8"))
PROFILES = json.loads((ROOT / "analysis_profiles.json").read_text(encoding="utf-8"))

st.set_page_config(page_title=f"Stake Multiagente v{APP_VERSION}", page_icon="🎯", layout="wide")
st.title("🎯 Corridas Stake — OpenRouter multiagente")
st.caption(
    f"Versión {APP_VERSION} · Stake directo · Perfiles especializados · "
    "Consenso de modelos · Juez independiente"
)

DEFAULT_SINGLE_MODEL = "google/gemini-3.8-flash"
DEFAULT_ANALYST_MODELS = [
    "google/gemini-3.8-flash",
    "qwen/qwen3.8-max-0902",
    "deepseek/deepseek-v4-pro",
]


def secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name)
        if value not in (None, "", "None", "null"):
            return str(value).strip()
    except Exception:
        pass
    value = os.getenv(name)
    return str(value).strip() if value else default


def openrouter_secret() -> str:
    direct = secret("OPENROUTER_API_KEY")
    if direct:
        return direct
    for section_name in ("openrouter", "OPENROUTER"):
        try:
            section = st.secrets[section_name]
            value = section.get("api_key") or section.get("API_KEY")
            if value:
                return str(value).strip()
        except Exception:
            continue
    return ""


@st.cache_data(ttl=3600, show_spinner=False)
def cached_models(api_key: str) -> list[dict]:
    return list_models(api_key)


@st.cache_data(ttl=300, show_spinner=False)
def cached_key_status(api_key: str) -> dict:
    return verify_api_key(api_key)


def display_table(rows: list[dict]) -> None:
    if not rows:
        st.info("No hay filas para mostrar.")
        return
    frame = pd.DataFrame(rows)
    # Mantener una fecha real para que el orden sea cronológico. Si se convierte
    # primero a texto con AM/PM, Streamlit la ordena alfabéticamente.
    if "start_rd" in frame.columns:
        frame["_start_sort"] = pd.to_datetime(
            frame["start_rd"], errors="coerce", utc=True
        )
        frame = frame.sort_values(
            ["_start_sort", "sport", "event"],
            ascending=[True, True, True],
            na_position="last",
            kind="stable",
        )
    columns = {
        "start_rd": "Hora RD", "sport": "Deporte", "event": "Evento",
        "market": "Mercado", "selection": "Selección", "odds": "Cuota",
        "stake_market_favorite": "Favorito Stake",
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
        view["Hora RD"] = pd.to_datetime(
            view["Hora RD"], errors="coerce", utc=True
        ).dt.tz_convert("America/Santo_Domingo")
    st.dataframe(
        view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Hora RD": st.column_config.DatetimeColumn(
                "Hora RD",
                help="Hora de inicio en República Dominicana (UTC−4).",
                format="DD/MM/YYYY hh:mm a",
            )
        },
    )


def history_context(history: list[dict]) -> str:
    safe = [
        {key: item.get(key) for key in ("event", "selection", "odds", "result", "note")}
        for item in history[-20:]
    ]
    return json.dumps(safe, ensure_ascii=False) if safe else ""


def minutes_until_start(start_value: str) -> float:
    """Minutos restantes usando el offset incluido por Stake."""
    try:
        start = datetime.fromisoformat(str(start_value).replace("Z", "+00:00"))
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        return (start.astimezone(timezone.utc) - datetime.now(timezone.utc)).total_seconds() / 60
    except (TypeError, ValueError):
        return float("-inf")


for key, default in {
    "scan": None,
    "evaluations": [],
    "history": [],
    "model_catalog": [],
    "scan_notice": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.header("Configuración")
    api_key_input = st.text_input("OpenRouter API key", type="password")
    api_key = api_key_input.strip() or openrouter_secret()
    if api_key:
        try:
            key_status = cached_key_status(api_key)
            key_label = key_status.get("label") or "clave reconocida"
            st.success(f"OpenRouter conectado: {key_label}")
        except OpenRouterError as exc:
            st.error(str(exc))
    else:
        st.error("No se encontró OPENROUTER_API_KEY en Secrets.")
    if st.button("Actualizar catálogo de modelos", use_container_width=True):
        cached_models.clear()
    try:
        st.session_state.model_catalog = cached_models(api_key)
    except Exception as exc:
        st.warning(f"Catálogo no disponible: {exc}")
    model_ids = [item["id"] for item in st.session_state.model_catalog]
    configured_default = secret("OPENROUTER_MODEL", DEFAULT_SINGLE_MODEL)
    priority_models = DEFAULT_ANALYST_MODELS + [configured_default]
    model_ids = list(dict.fromkeys(priority_models + model_ids))

    mode = st.radio("Modo de análisis", ["Consenso multiagente", "Modelo único"])
    if mode == "Consenso multiagente":
        analyst_models = st.multiselect(
            "Modelos analistas (2–3)", model_ids,
            default=DEFAULT_ANALYST_MODELS,
            max_selections=3,
        )
        judge_model = st.selectbox(
            "Modelo juez", model_ids, index=model_ids.index(DEFAULT_SINGLE_MODEL)
        )
    else:
        analyst_models = [st.selectbox(
            "Modelo analista", model_ids, index=model_ids.index(DEFAULT_SINGLE_MODEL)
        )]
        judge_model = analyst_models[0]

    profile_name = st.selectbox("Perfil de análisis", list(PROFILES))
    profile = PROFILES[profile_name]
    st.caption(profile["description"])
    use_web = st.checkbox("Búsqueda web de OpenRouter", value=True)
    bankroll = st.number_input("Banca actual (USD)", min_value=0.0, value=151.04, step=1.0)
    pending = st.number_input("Exposición pendiente (USD)", min_value=0.0, value=0.0, step=1.0)
    hours = st.slider("Ventana futura (horas)", 2, 36, 18)
    min_lead_minutes = st.slider(
        "Anticipación mínima al comenzar (minutos)",
        5,
        90,
        30,
        5,
        help="Descarta eventos demasiado cercanos para completar el consenso antes del inicio.",
    )
    max_candidates = st.slider("Candidatos a validar", 1, 10, 2)
    calls = max_candidates * len(analyst_models) + (max_candidates if mode == "Consenso multiagente" else 0)
    st.info(
        f"La corrida usará aproximadamente {calls} llamadas a OpenRouter. "
        "Los analistas y los jueces se ejecutan en paralelo."
    )
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
    if st.session_state.scan_notice:
        st.success(st.session_state.scan_notice)
        st.session_state.scan_notice = ""
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
                [
                    row
                    for row in scan["candidates"]
                    if row["market_overround"] <= CONFIG["market_quality"]["max_overround"]
                    and minutes_until_start(row["start_rd"]) >= min_lead_minutes
                ],
                key=lambda row: (
                    row["start_rd"],
                    -row["market_no_vig_probability"],
                    row["sport"],
                    row["event"],
                ),
            )
            st.session_state.scan = scan
            st.session_state.evaluations = []
            st.session_state.scan_notice = "Consulta directa terminada. Ya puedes ejecutar el análisis."
            progress_bar.empty()
            st.rerun()
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
            eligible, results = [], []
            for candidate in targets:
                if minutes_until_start(candidate["start_rd"]) < min_lead_minutes:
                    result_row = evaluate(
                        candidate, None, CONFIG, bankroll, pending, profile=profile
                    )
                    result_row.setdefault("reasons", []).append(
                        f"Quedan menos de {min_lead_minutes} minutos para comenzar."
                    )
                    result_row["status"] = "Descartado"
                    results.append(result_row)
                else:
                    eligible.append(candidate)

            analyst_steps = len(eligible) * len(analyst_models)
            judge_steps = len(eligible) if mode == "Consenso multiagente" else 0
            total_steps = analyst_steps + judge_steps
            completed = 0
            bar = st.progress(0.0, text="Ejecutando analistas en paralelo…")
            context = history_context(st.session_state.history)

            # Todas las combinaciones candidato-modelo se envían al mismo tiempo.
            # El límite evita saturar OpenRouter cuando se seleccionan muchos eventos.
            analyses_by_candidate = {index: {} for index in range(len(eligible))}
            errors_by_candidate = {index: [] for index in range(len(eligible))}
            max_parallel = min(12, max(1, analyst_steps))
            with ThreadPoolExecutor(max_workers=max_parallel) as executor:
                future_models = {}
                for index, candidate in enumerate(eligible):
                    for analyst_model in analyst_models:
                        future = executor.submit(
                            validate_candidate,
                            candidate,
                            api_key,
                            analyst_model,
                            use_web,
                            90,
                            profile_name,
                            profile["instructions"],
                            context,
                        )
                        future_models[future] = (index, analyst_model)
                for future in as_completed(future_models):
                    index, analyst_model = future_models[future]
                    candidate = eligible[index]
                    try:
                        analyses_by_candidate[index][analyst_model] = future.result()
                    except (OpenRouterError, ValueError, TypeError, AttributeError) as exc:
                        errors_by_candidate[index].append(f"{analyst_model}: {exc}")
                    except Exception as exc:
                        errors_by_candidate[index].append(
                            f"{analyst_model}: error inesperado del proveedor "
                            f"({type(exc).__name__})."
                        )
                    completed += 1
                    bar.progress(
                        completed / max(total_steps, 1),
                        text=f"Analista completado: {candidate['selection']}",
                    )

            prepared = []
            for index, candidate in enumerate(eligible):
                analyses = [
                    analyses_by_candidate[index][model]
                    for model in analyst_models
                    if model in analyses_by_candidate[index]
                ]
                errors = errors_by_candidate[index]
                model_result = None
                try:
                    if mode == "Consenso multiagente":
                        model_result = build_consensus(analyses, profile["max_model_spread"])
                    elif analyses:
                        model_result = analyses[0]
                except (OpenRouterError, ValueError) as exc:
                    model_result = None
                    errors.append(str(exc))
                prepared.append((candidate, analyses, errors, model_result))

            judges: dict[int, dict] = {}
            if mode == "Consenso multiagente":
                judge_jobs = {
                    index: item for index, item in enumerate(prepared) if item[3]
                }
                with ThreadPoolExecutor(max_workers=min(8, max(1, len(judge_jobs)))) as executor:
                    futures = {
                        executor.submit(
                            judge_consensus,
                            item[0], item[3], api_key, judge_model, profile_name,
                        ): index
                        for index, item in judge_jobs.items()
                    }
                    for future in as_completed(futures):
                        index = futures[future]
                        try:
                            judges[index] = future.result()
                        except Exception as exc:
                            prepared[index][2].append(
                                f"Juez: {exc}" if isinstance(exc, OpenRouterError)
                                else f"Juez: error inesperado ({type(exc).__name__})."
                            )
                        completed += 1
                        bar.progress(
                            completed / max(total_steps, 1),
                            text=f"Juez completado: {prepared[index][0]['selection']}",
                        )

            for index, (candidate, analyses, errors, model_result) in enumerate(prepared):
                judge = judges.get(index)
                result_row = evaluate(
                    candidate, model_result, CONFIG, bankroll, pending,
                    profile=profile, judge=judge,
                )
                if errors:
                    result_row.setdefault("reasons", []).extend(errors)
                    if result_row.get("status") == "Aprobado":
                        result_row["status"] = "Descartado"
                if model_result:
                    result_row["model_spread"] = model_result.get("model_spread")
                    result_row["individual_analyses"] = model_result.get("individual_analyses", analyses)
                remaining = minutes_until_start(candidate["start_rd"])
                if remaining <= 0:
                    result_row["status"] = "Descartado"
                    result_row.setdefault("reasons", []).append(
                        "El evento comenzó mientras se ejecutaba el análisis."
                    )
                elif remaining < 5:
                    result_row["status"] = "Descartado"
                    result_row.setdefault("reasons", []).append(
                        "Quedan menos de 5 minutos: no hay tiempo operativo para apostar."
                    )
                results.append(result_row)
            st.session_state.evaluations = rank_candidates(
                block_opposing_approvals(results)
            )
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
