"""Staple GraphQL client.

Three non-obvious properties of this server drive the shape of this module
(SPEC §16):

  1. Auth failures return HTTP 200. @authenticate is a per-field directive that
     throws inside the resolver, so the transport still succeeds. Checking the
     status code alone lets expired tokens through as empty candidate lists.
  2. The "expired" message lies — sending no token at all returns
     "Token has expired", not "token missing". Never branch on message text.
  3. Partial failure is possible: some fields populated, some null, plus errors.
"""

import httpx

from search_agent.config import settings


class GraphQLError(Exception):
    def __init__(self, errors):
        self.errors = errors
        super().__init__(str(errors))


class UnauthenticatedError(GraphQLError):
    """Token missing, malformed, expired, or forged. Always surface as HTTP 401."""


async def gql(authorization: str, query: str, variables: dict | None = None) -> dict:
    """Execute a query against Staple with the caller's bearer token.

    `authorization` is the raw header value, including the "Bearer " prefix —
    Staple's authenticate() splits on a space and rejects anything else.
    """
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            settings.staple_graphql_url,
            json={"query": query, "variables": variables or {}},
            headers={"Authorization": authorization, "Content-Type": "application/json"},
        )

    body = response.json()

    # Note the order: inspect `errors` BEFORE trusting the status code.
    errors = body.get("errors")
    if errors:
        codes = {
            e.get("extensions", {}).get("code")
            for e in errors
            if isinstance(e, dict)
        }
        if "UNAUTHENTICATED" in codes:
            raise UnauthenticatedError(errors)
        raise GraphQLError(errors)

    return body
