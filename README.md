# Stake Direct v3 multiagente — Streamlit + OpenRouter

Aplicación para consultar cuotas prepartido directamente desde
`https://odds-data.stake.com`, filtrar candidatos y validarlos mediante un segundo
modelo disponible en OpenRouter.

La interfaz carga el catálogo actual de OpenRouter y permite seleccionar un modelo
individual o entre dos y tres modelos analistas más un modelo juez independiente.

## Funciones

- Consulta directa de deportes, categorías, torneos, eventos y mercados de Stake.
- Horario fijo de República Dominicana (UTC−4).
- Descarta eventos iniciados, simulaciones, juveniles, reservas y amistosos.
- Rango de cuotas configurable mediante `stake_logic_v2_strict.json` (1.40–2.00).
- Cálculo de probabilidad implícita, probabilidad sin margen y overround.
- Validación externa mediante OpenRouter, opcionalmente con búsqueda web.
- Selector dinámico de cualquier modelo disponible en OpenRouter.
- Consenso de 2–3 modelos con límite de dispersión probabilística.
- Modelo juez que audita evidencia y contradicciones antes de aprobar.
- Perfiles: estricto, equilibrado, MLB, fútbol y hockey.
- Historial importable/exportable de resultados y correcciones.
- Gates estrictos: EV mínimo +4%, divergencia máxima 9 puntos, fuentes y roster.
- Límites de stake y exposición de banca.
- Resultado separado en aprobado, pendiente y descartado.
- Exportación completa en JSON.

## Archivos para Streamlit Cloud

- `app.py`: interfaz principal.
- `stake_client.py`: cliente directo de Stake y extracción de mercados.
- `openrouter_client.py`: segundo modelo y validación estructurada.
- `analysis_profiles.json`: perfiles, umbrales e instrucciones especializadas.
- `engine.py`: reglas, EV, divergencia, empate y riesgo.
- `stake_logic_v2_strict.json`: configuración editable.
- `requirements.txt`: dependencias.
- `.streamlit/config.toml`: configuración visual.
- `.streamlit/secrets.toml.example`: ejemplo de secretos; no publiques una clave real.

## Ejecución local

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edita secrets.toml y coloca tu clave real
streamlit run app.py
```

## Despliegue en Streamlit Community Cloud

1. Sube el contenido de esta carpeta a un repositorio privado o público.
2. En Streamlit Community Cloud selecciona el repositorio y `app.py`.
3. Abre **Settings > Secrets** y agrega:

```toml
OPENROUTER_API_KEY = "sk-or-v1-tu-clave"
OPENROUTER_MODEL = "openai/gpt-4.1-mini"
```

4. Pulsa **Deploy**.

No subas `.streamlit/secrets.toml` al repositorio. El `.gitignore` ya lo excluye.

## Uso recomendado

1. Introduce banca y exposición pendiente.
2. Selecciona la ventana horaria.
3. Elige **Modelo único** o **Consenso multiagente**.
4. Selecciona cualquier modelo del catálogo de OpenRouter y el perfil deportivo.
5. Pulsa **Consultar Stake** y luego **Ejecutar análisis**.
6. En consenso, cada modelo analiza por separado y el juez revisa la conclusión.
7. Solo una selección que supere modelos, juez y gates aparecerá aprobada.

## Resultados y correcciones

La pestaña correspondiente permite registrar ganados, perdidos, void y notas. Ese
historial se entrega a los modelos únicamente como contexto de calibración. Descarga
`historial_stake.json` para conservarlo y reimpórtalo después de un reinicio; el disco
de Streamlit Community Cloud no debe tratarse como almacenamiento permanente.

## Coste de OpenRouter

La barra lateral muestra cuántas llamadas aproximadas ejecutará la corrida. Con cuatro
candidatos, dos analistas y un juez son doce llamadas. El coste depende de los modelos
seleccionados y de la búsqueda web. Puedes usar el modo de modelo único para reducirlo.

## Consideraciones

- El plugin web de OpenRouter depende de la compatibilidad del modelo/proveedor.
- Si no hay fuentes, roster confirmado o JSON válido, la selección no se aprueba.
- El EV estándar usa `p × cuota − 1`. Para DNB la app calcula explícitamente
  `P(victoria) × cuota + P(empate) − 1` y compara contra la probabilidad condicional
  sin empate del mercado.
- Stake puede modificar su API pública sin aviso; los errores se muestran en pantalla.
- La app analiza, pero no inicia sesión en Stake ni coloca apuestas.
