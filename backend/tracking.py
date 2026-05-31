from __future__ import annotations

import asyncio
import logging
import time

import httpx

from .config import get_settings
from .schemas import Aircraft

logger = logging.getLogger("geoint.tracking")

OPENSKY_TOKEN_URL = (
    "https://auth.opensky-network.org/auth/realms/opensky-network/"
    "protocol/openid-connect/token"
)
OPENSKY_STATES_URL = "https://opensky-network.org/api/states/all"


class OpenSkyClient:
    """Fetches live aircraft state vectors from the OpenSky Network.

    Uses OAuth2 client credentials when configured, otherwise falls back to
    anonymous (heavily rate-limited) access.
    """

    def __init__(self) -> None:
        self._token: str | None = None
        self._token_expiry: float = 0.0

    async def _get_token(self, client: httpx.AsyncClient) -> str | None:
        s = get_settings()
        if not (s.opensky_client_id and s.opensky_client_secret):
            return None
        if self._token and time.time() < self._token_expiry - 30:
            return self._token
        resp = await client.post(
            OPENSKY_TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": s.opensky_client_id,
                "client_secret": s.opensky_client_secret,
            },
        )
        resp.raise_for_status()
        payload = resp.json()
        self._token = payload["access_token"]
        self._token_expiry = time.time() + payload.get("expires_in", 1800)
        return self._token

    async def fetch(self) -> list[Aircraft]:
        s = get_settings()
        min_lat, max_lat, min_lon, max_lon = s.bbox
        params = {
            "lamin": min_lat,
            "lamax": max_lat,
            "lomin": min_lon,
            "lomax": max_lon,
        }
        async with httpx.AsyncClient(timeout=15) as client:
            headers = {}
            token = await self._get_token(client)
            if token:
                headers["Authorization"] = f"Bearer {token}"
            resp = await client.get(OPENSKY_STATES_URL, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        return _parse_states(data.get("states") or [])


def _parse_states(states: list[list]) -> list[Aircraft]:
    """Map OpenSky state-vector arrays into Aircraft models."""
    aircraft: list[Aircraft] = []
    for st in states:
        try:
            aircraft.append(
                Aircraft(
                    icao24=st[0],
                    callsign=(st[1] or "").strip() or None,
                    lon=st[5],
                    lat=st[6],
                    altitude_m=st[7] if st[7] is not None else st[13],
                    on_ground=bool(st[8]),
                    velocity_ms=st[9],
                    heading_deg=st[10],
                    last_contact=st[4],
                )
            )
        except (IndexError, TypeError):
            continue
    return [a for a in aircraft if a.lat is not None and a.lon is not None]


class TrackHub:
    """Polls OpenSky on an interval and broadcasts to WebSocket subscribers."""

    def __init__(self) -> None:
        self._clients: set = set()
        self._latest: list[Aircraft] = []
        self._task: asyncio.Task | None = None
        self._client = OpenSkyClient()
        self.poll_count = 0

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            self._task = None

    async def register(self, ws) -> None:
        self._clients.add(ws)
        # Send a snapshot immediately so the map populates on connect.
        await ws.send_json(
            {"type": "tracks", "aircraft": [a.model_dump() for a in self._latest]}
        )

    def unregister(self, ws) -> None:
        self._clients.discard(ws)

    @property
    def latest(self) -> list[Aircraft]:
        return self._latest

    async def _loop(self) -> None:
        s = get_settings()
        while True:
            try:
                self._latest = await self._client.fetch()
                self.poll_count += 1
                await self._broadcast()
            except Exception as exc:  # noqa: BLE001 - keep the poll loop alive
                logger.warning("OpenSky poll failed: %s", exc)
            await asyncio.sleep(s.track_poll_seconds)

    async def _broadcast(self) -> None:
        payload = {
            "type": "tracks",
            "aircraft": [a.model_dump() for a in self._latest],
        }
        dead = []
        for ws in self._clients:
            try:
                await ws.send_json(payload)
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)


hub = TrackHub()
