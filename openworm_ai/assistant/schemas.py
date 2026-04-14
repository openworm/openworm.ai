"""
State and routing schemas for the OpenWorm assistant.

AssistantState uses TypedDict (not Pydantic BaseModel) because LangGraph
recommends TypedDict for graph state when running inside Streamlit.
Streamlit re-executes the script on every interaction, re-importing all
modules and creating fresh class definitions.  @st.cache_resource keeps
the compiled graph from a previous run, so Pydantic's strict class-identity
checks reject structurally-identical objects from the reloaded module.
TypedDict has no runtime validation, so no class-identity mismatch.

See: https://github.com/streamlit/streamlit/issues/6765
     https://github.com/langchain-ai/langgraph/issues/5733

QueryTypeSchema remains a Pydantic BaseModel because it is only used
transiently inside the classifier node (for LLM structured output parsing)
and its .query_type string is what gets stored in state.
"""

from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field
from typing import TypedDict
from typing_extensions import Dict, List, Literal, Tuple


class QueryTypeSchema(BaseModel):
    """LLM classifier output — question or task."""

    query_type: Literal["undefined", "question", "task"] = Field(
        default="undefined",
    )


class AssistantState(TypedDict, total=False):
    query: str
    query_type: str  # "undefined", "question", or "task"
    messages: List[AnyMessage]
    context_summary: str
    message_for_user: str
    # Retrieved corpus references: {query: [(Document, score), ...]}
    reference_material: Dict[str, List[Tuple]]
    # Base64-encoded plot image from tool execution (e.g. HH voltage trace)
    plot_base64: str
    # Voltage trace data for client-side rendering (fallback when sandbox
    # matplotlib is unavailable). Dict with keys: t_ms, v_mv (lists of floats)
    plot_data: dict
