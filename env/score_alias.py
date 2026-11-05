"""ASGI middleware: serve Azure ML's /score path with vLLM's /classify handler.

The managed-endpoint gateway does not always rewrite /score to the environment's scoring_route,
so the container accepts both. Loaded via `vllm serve --middleware score_alias.ScoreAlias`.
"""


class ScoreAlias:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("path", "").rstrip("/") == "/score":
            scope = dict(scope, path="/classify", raw_path=b"/classify")
        await self.app(scope, receive, send)
