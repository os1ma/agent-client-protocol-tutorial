import asyncio
import os
from typing import Any
from uuid import uuid4

from acp import (
    Agent,
    InitializeResponse,
    NewSessionResponse,
    PromptResponse,
    run_agent,
    text_block,
    update_agent_message,
)
from acp.interfaces import Client
from acp.schema import (
    AudioContentBlock,
    ClientCapabilities,
    EmbeddedResourceContentBlock,
    HttpMcpServer,
    ImageContentBlock,
    Implementation,
    McpServerStdio,
    ResourceContentBlock,
    SseMcpServer,
    TextContentBlock,
)
from dotenv import load_dotenv
from openai import OpenAI


class OpenAIAgent(Agent):
    _conn: Client

    def on_connect(self, conn: Client) -> None:
        self._conn = conn

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        return InitializeResponse(protocol_version=protocol_version)

    async def new_session(
        self,
        cwd: str,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio],
        **kwargs: Any,
    ) -> NewSessionResponse:
        return NewSessionResponse(session_id=uuid4().hex)

    async def prompt(
        self,
        prompt: list[
            TextContentBlock
            | ImageContentBlock
            | AudioContentBlock
            | ResourceContentBlock
            | EmbeddedResourceContentBlock
        ],
        session_id: str,
        **kwargs: Any,
    ) -> PromptResponse:
        for block in prompt:
            text = (
                block.get("text", "")
                if isinstance(block, dict)
                else getattr(block, "text", "")
            )

            openci_client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
            )

            response = openci_client.responses.create(
                model="gpt-5-nano",
                instructions="You are a coding assistant that talks like a pirate.",
                input=text,
            )

            chunk = update_agent_message(text_block(response.output_text))
            chunk.field_meta = {"echo": True}
            chunk.content.field_meta = {"echo": True}

            await self._conn.session_update(
                session_id=session_id,
                update=chunk,
                source="echo_agent",
            )
        return PromptResponse(stop_reason="end_turn")


async def main() -> None:
    load_dotenv()
    await run_agent(OpenAIAgent())


if __name__ == "__main__":
    asyncio.run(main())
