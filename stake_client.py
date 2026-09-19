from __future__ import annotations

import concurrent.futures as cf
import re
import time
import urllib.parse
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import requests

BASE_URL = "https://odds-data.stake.com"
RD = timezone(timedelta(hours=-4), name="AST")


class StakeError(RuntimeError):
    pass


class StakeClient:
    def __init__(self, timeout: int = 18, workers: int = 32):
        self.timeout = timeout
        self.workers = workers
        self.headers = {
            "Accept": "application/json",
            "User-Agent": "stake-direct-streamlit-v2/2.1",
        }

    def get(self, path: str) -> Any:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = requests.get(
                    BASE_URL + path, headers=self.headers, timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(0.6 * (attempt + 1))
        raise StakeError(f"Stake no respondió correctamente en {path}: {last_error}")

    @staticmethod
    def _slug(value: Any) -> str:
        return urllib.parse.quote(str(value), safe="-")

    @staticmethod
    def _stamp(value: Any) -> datetime | None:
        if isinstance(value, (int, float)):
            seconds = value / 1000 if value > 10_000_000_000 else value
            return datetime.fromtimestamp(seconds, timezone.utc)
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(
                timezone.utc
            )
        except (TypeError, ValueError):
            return None

    def _parallel(
        self,
        jobs: list[tuple[str, Any]],
        progress: Callable[[int, int], None] | None = None,
    ) -> tuple[list[tuple[Any, Any]], int]:
        output: list[tuple[Any, Any]] = []
        errors = 0
        total = len(jobs)
        done = 0
        with cf.ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(self.get, path): context for path, context in jobs}
            for future in cf.as_completed(futures):
                done += 1
                try:
                    output.append((futures[future], future.result()))
                except Exception:
                    errors += 1
                if progress:
                    progress(done, total)
        return output, errors

    def scan(
        self,
        config: dict[str, Any],
        hours_ahead: int = 24,
        selected_sports: list[str] | None = None,
        progress: Callable[[str, float], None] | None = None,
    ) -> dict[str, Any]:
        now = datetime.now(RD)
        end = now + timedelta(hours=hours_ahead)
        virtual = re.compile(config["exclusions"]["sport_or_event_regex"], re.I)
        fragile = re.compile(config["exclusions"]["competition_regex"], re.I)
        primary = re.compile(config["markets"]["primary_regex"], re.I)
        protected = re.compile(config["markets"]["protected_regex"], re.I)
        odds_min, odds_max = config["odds_range"]
        failures = 0

        def update(label: str, fraction: float) -> None:
            if progress:
                progress(label, min(max(fraction, 0.0), 1.0))

        update("Consultando deportes activos", 0.02)
        sports = [
            sport
            for sport in self.get("/sports")
            if sport.get("slug")
            and sport.get("enabled", True)
            and not virtual.search(f"{sport.get('name', '')} {sport.get('slug', '')}")
        ]
        if selected_sports:
            wanted = {item.lower() for item in selected_sports}
            sports = [item for item in sports if item["slug"].lower() in wanted]

        categories: list[tuple[dict, dict]] = []
        rows, errors = self._parallel(
            [(f"/sports/{self._slug(s['slug'])}/categories", s) for s in sports],
            lambda d, t: update("Consultando categorías", 0.05 + 0.10 * d / max(t, 1)),
        )
        failures += errors
        for sport, payload in rows:
            for category in payload.get("categories", []):
                if category.get("slug") and category.get("enabled", True):
                    categories.append((sport, category))

        tournaments: list[tuple[dict, dict, dict]] = []
        rows, errors = self._parallel(
            [
                (
                    f"/sports/{self._slug(s['slug'])}/{self._slug(c['slug'])}/tournaments",
                    (s, c),
                )
                for s, c in categories
            ],
            lambda d, t: update("Consultando torneos", 0.15 + 0.18 * d / max(t, 1)),
        )
        failures += errors
        for (sport, category), payload in rows:
            for tournament in payload.get("tournaments", []):
                descriptor = f"{tournament.get('name', '')} {tournament.get('slug', '')}"
                if (
                    tournament.get("slug")
                    and tournament.get("enabled", True)
                    and not fragile.search(descriptor)
                ):
                    tournaments.append((sport, category, tournament))

        fixtures: dict[str, tuple[dict, dict, dict, dict]] = {}
        rows, errors = self._parallel(
            [
                (
                    f"/sports/{self._slug(s['slug'])}/{self._slug(c['slug'])}/{self._slug(t['slug'])}/fixtures",
                    (s, c, t),
                )
                for s, c, t in tournaments
            ],
            lambda d, t: update("Consultando eventos", 0.33 + 0.27 * d / max(t, 1)),
        )
        failures += errors
        blocked_status = {"live", "ended", "inactive", "inplay", "in-play"}
        for (sport, category, tournament), payload in rows:
            for fixture in payload.get("fixtures", []):
                start = self._stamp(fixture.get("startTime") or fixture.get("date"))
                descriptor = f"{fixture.get('name', '')} {fixture.get('slug', '')}"
                status = str(fixture.get("status", "")).lower()
                if (
                    start
                    and now <= start.astimezone(RD) <= end
                    and fixture.get("preMatchEnabled", True)
                    and fixture.get("enabled", True)
                    and status not in blocked_status
                    and fixture.get("slug")
                    and not virtual.search(descriptor)
                    and not fragile.search(descriptor)
                ):
                    key = str(fixture.get("id") or fixture["slug"])
                    fixtures[key] = (sport, category, tournament, fixture)

        candidates: list[dict[str, Any]] = []
        rows, errors = self._parallel(
            [
                (f"/fixtures/{self._slug(context[3]['slug'])}", context)
                for context in fixtures.values()
            ],
            lambda d, t: update("Descargando mercados", 0.60 + 0.38 * d / max(t, 1)),
        )
        failures += errors
        for (sport, _category, tournament, fixture), payload in rows:
            names = fixture.get("competitors") or []
            if names and isinstance(names[0], dict):
                names = [item.get("name", "") for item in names]
            if len(names) < 2:
                names = re.split(r"\s+-\s+", fixture.get("name", ""), maxsplit=1)
            event = " vs ".join(map(str, names[:2]))
            seen: set[str] = set()
            for group in payload.get("groups", []):
                for bundle in group.get("markets", []):
                    for market in bundle if isinstance(bundle, list) else [bundle]:
                        if not isinstance(market, dict):
                            continue
                        market_name = str(market.get("name", "")).strip()
                        marker = str(market.get("id") or market_name)
                        if (
                            not primary.match(market_name)
                            or marker in seen
                            or str(market.get("status", "active")).lower()
                            not in {"active", "open"}
                        ):
                            continue
                        seen.add(marker)
                        outcomes = []
                        for outcome in market.get("outcomes", []):
                            try:
                                odd = float(outcome.get("odds"))
                            except (TypeError, ValueError):
                                continue
                            if outcome.get("active", True) and odd > 1:
                                outcomes.append(
                                    {"selection": str(outcome.get("name")), "odds": odd}
                                )
                        if len(outcomes) < 2:
                            continue
                        inverse_sum = sum(1 / item["odds"] for item in outcomes)
                        overround = inverse_sum - 1
                        for outcome in outcomes:
                            if odds_min <= outcome["odds"] <= odds_max:
                                implied = 1 / outcome["odds"]
                                candidates.append(
                                    {
                                        "sport": sport["slug"],
                                        "league": tournament.get("name", ""),
                                        "event": event,
                                        "start_rd": self._stamp(fixture["startTime"])
                                        .astimezone(RD)
                                        .isoformat(),
                                        "market": market_name,
                                        **outcome,
                                        "implied_probability": round(implied, 4),
                                        "market_no_vig_probability": round(
                                            implied / inverse_sum, 4
                                        ),
                                        "market_overround": round(overround, 4),
                                        "protected_market": bool(
                                            protected.search(market_name)
                                        ),
                                    }
                                )
        update("Consulta Stake terminada", 1.0)
        candidates.sort(key=lambda row: row["start_rd"])
        return {
            "source": BASE_URL,
            "generated_rd": now.isoformat(),
            "sports": len(sports),
            "tournaments": len(tournaments),
            "fixtures_remaining": len(fixtures),
            "request_failures": failures,
            "candidates": candidates,
        }
