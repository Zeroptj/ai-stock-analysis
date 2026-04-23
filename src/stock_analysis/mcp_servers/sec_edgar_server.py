"""SEC EDGAR MCP server — exposes SEC filings + XBRL facts as MCP tools.

Transport: stdio. Launched by an MCP-capable client (Claude Desktop,
Claude Code, or our own orchestrator via the mcp client library).

Tools:
    - get_company_facts(ticker): XBRL financial facts (revenue, net income, ...)
    - get_financial_summary(ticker, years=5): distilled annual financials
    - get_filings(ticker, form_type, count): list recent filings by form type
    - get_proxy_statements(ticker, count=2, include_text=True): DEF 14A with full text
    - get_material_events(ticker, count=5, include_text=True): 8-K with latest text
    - get_filing_text(document_url): download + plain-text extract

Run standalone for debugging:
    python -m stock_analysis.mcp_servers.sec_edgar_server
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from stock_analysis.data.sec_edgar_client import (
    _extract_financial_facts,
    fetch_filing_text,
    get_company_facts,
    get_filing_list,
)

server = Server("sec-edgar")


def _dumps(obj: Any) -> str:
    import json
    return json.dumps(obj, indent=2, default=str)


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_company_facts",
            description=(
                "Fetch raw XBRL company facts (all reported concepts) for a US-listed "
                "ticker from SEC EDGAR. Large payload — prefer get_financial_summary "
                "for modeling."
            ),
            inputSchema={
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        ),
        Tool(
            name="get_financial_summary",
            description=(
                "Return the last N annual values for key line items: revenue, net "
                "income, total assets/liabilities, operating cash flow, capex, EPS, "
                "etc. Suitable for DCF / ratio analysis."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "years": {"type": "integer", "default": 5},
                },
                "required": ["ticker"],
            },
        ),
        Tool(
            name="get_filings",
            description=(
                "List recent filings of a given form type "
                "(10-K, 10-Q, DEF 14A, 8-K, Form 4, etc.). Returns metadata + URL."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "form_type": {"type": "string"},
                    "count": {"type": "integer", "default": 5},
                },
                "required": ["ticker", "form_type"],
            },
        ),
        Tool(
            name="get_proxy_statements",
            description=(
                "Fetch recent DEF 14A proxy statements (shareholder meeting agenda, "
                "voting proposals, executive compensation). Set include_text=true "
                "to download the latest filing as plain text."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "count": {"type": "integer", "default": 2},
                    "include_text": {"type": "boolean", "default": True},
                },
                "required": ["ticker"],
            },
        ),
        Tool(
            name="get_material_events",
            description=(
                "Fetch recent 8-K filings (earnings announcements, M&A, director "
                "changes, other material events). Set include_text=true to "
                "download the latest event text."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "count": {"type": "integer", "default": 5},
                    "include_text": {"type": "boolean", "default": True},
                },
                "required": ["ticker"],
            },
        ),
        Tool(
            name="get_filing_text",
            description=(
                "Download a filing's primary document and return plain text "
                "(HTML stripped, truncated to ~50K chars)."
            ),
            inputSchema={
                "type": "object",
                "properties": {"document_url": {"type": "string"}},
                "required": ["document_url"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    async with httpx.AsyncClient(timeout=60.0) as client:
        if name == "get_company_facts":
            facts = await get_company_facts(arguments["ticker"], client, use_cache=False)
            return [TextContent(type="text", text=_dumps({
                "entity": facts.get("entityName"),
                "cik": facts.get("cik"),
                "concept_count": len(facts.get("facts", {}).get("us-gaap", {})),
            }))]

        if name == "get_financial_summary":
            facts = await get_company_facts(arguments["ticker"], client, use_cache=False)
            summary = _extract_financial_facts(facts)
            years = int(arguments.get("years", 5))
            trimmed = {k: v[:years] for k, v in summary.items()}
            return [TextContent(type="text", text=_dumps({
                "ticker": arguments["ticker"].upper(),
                "entity": facts.get("entityName"),
                "financials": trimmed,
            }))]

        if name == "get_filings":
            filings = await get_filing_list(
                arguments["ticker"], client,
                filing_type=arguments["form_type"],
                count=int(arguments.get("count", 5)),
            )
            return [TextContent(type="text", text=_dumps(filings))]

        if name == "get_proxy_statements":
            filings = await get_filing_list(
                arguments["ticker"], client,
                filing_type="DEF 14A",
                count=int(arguments.get("count", 2)),
            )
            if filings and arguments.get("include_text", True):
                filings[0]["text"] = await fetch_filing_text(filings[0], client)
            return [TextContent(type="text", text=_dumps(filings))]

        if name == "get_material_events":
            filings = await get_filing_list(
                arguments["ticker"], client,
                filing_type="8-K",
                count=int(arguments.get("count", 5)),
            )
            if filings and arguments.get("include_text", True):
                filings[0]["text"] = await fetch_filing_text(filings[0], client)
            return [TextContent(type="text", text=_dumps(filings))]

        if name == "get_filing_text":
            text = await fetch_filing_text(
                {"document_url": arguments["document_url"]}, client
            )
            return [TextContent(type="text", text=text)]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _run() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
