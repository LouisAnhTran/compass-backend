"""Token → verified user id (SPEC §2.3).

Compass never holds JWT_SECRET_KEY, so it cannot verify a signature itself.
It therefore MUST NOT read `uid` out of the JWT payload: the claim is right
there (Staple signs {uid, identity}) and decoding it is one line, but nothing
in that path checks the signature. A caller can hand Compass {"uid": 1} with
garbage after the dot and read another user's conversations — Staple would
reject such a token, but Staple is never in the path for GET /conversations.

Instead we ask the service that does hold the secret.
"""

import time

from fastapi import HTTPException

from search_agent.clients.graphql import UnauthenticatedError, gql

PERSONAL_INFO = "query { getPersonalInfo { id } }"

# Keyed on the token string, never on thread_id — keying on thread_id would let
# a second caller inherit the first caller's identity. TTL stays well under the
# 10h token life so a revoked session cannot linger.
_CACHE: dict[str, tuple[int, float]] = {}
_TTL_SECONDS = 300


async def resolve_user_id(authorization: str | None) -> int:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token")

    cached = _CACHE.get(authorization)
    if cached and cached[1] > time.monotonic():
        return cached[0]

    try:
        body = await gql(authorization, PERSONAL_INFO)
    except UnauthenticatedError:
        raise HTTPException(401, "invalid or expired token")

    user_id = body["data"]["getPersonalInfo"]["id"]
    _CACHE[authorization] = (user_id, time.monotonic() + _TTL_SECONDS)
    return user_id
