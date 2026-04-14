"""
OpenWorm Assistant — mirrors NML_Assistant structure from neuroml-ai
but with C. elegans classifier prompt and MCP tool-use node replacing
the code stub.

Based on: neuroml_ai/neuroml_ai/assistant.py (Ankur Sinha, 2025)

Differences from NML_Assistant:
  - AssistantState is a TypedDict (not BaseModel) for Streamlit compat
    → state access uses state["key"] / state.get("key") instead of
      state.attribute
  - query_type stored as a plain string in state (not QueryTypeSchema)
    → QueryTypeSchema is still used for LLM structured output, but only
      the .query_type string is written to state
  - _tool_node replaces the _code_node stub with MCP tool execution
  - Classifier prompt is C. elegans / OpenWorm specific
"""

import json
import logging
import os
import sys
from textwrap import dedent

from gen_rag.rag import RAG
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from neuroml_ai_utils.llm import (
    add_memory_to_prompt,
    parse_output_with_thought,
    setup_llm,
)
from neuroml_ai_utils.logging import (
    LoggerInfoFilter,
    LoggerNotInfoFilter,
    logger_formatter_info,
    logger_formatter_other,
)

from .schemas import AssistantState, QueryTypeSchema


def _create_mcp_server():
    """Create the in-process MCP server with all OpenWorm tools registered."""
    from fastmcp import FastMCP
    from openworm_mcp.tools import hh_tools, wormbase_tools
    from openworm_mcp.utils import register_tools

    mcp = FastMCP("openworm_MCP", instructions="OpenWorm assistant tools")
    register_tools(mcp, [hh_tools, wormbase_tools])
    return mcp


class OpenWormAssistant(object):
    """OpenWorm Assistant class"""

    def __init__(
        self,
        vs_config_file: str,
        chat_model: str,
        mcp_server=None,
        logging_level: int = logging.DEBUG,
    ):
        self.chat_model = chat_model
        self.model = None

        self.vs_config_file = vs_config_file
        # Use in-process MCP server if none provided
        self.mcp_server = mcp_server or _create_mcp_server()

        self.checkpointer = InMemorySaver()

        # number of conversations after which to summarise
        # no need to summarise after each
        # 5 rounds: 10 messages
        self.num_recent_messages = 10

        self.logger = logging.getLogger("OpenWormAssistant")
        self.logger.setLevel(logging_level)
        self.logger.propagate = False

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.addFilter(LoggerInfoFilter())
        stdout_handler.setFormatter(logger_formatter_info)
        self.logger.addHandler(stdout_handler)

        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging_level)
        stderr_handler.addFilter(LoggerNotInfoFilter())
        stderr_handler.setFormatter(logger_formatter_other)
        self.logger.addHandler(stderr_handler)

    # -- NML_Assistant: returns QueryTypeSchema()
    # -- Here: returns "undefined" string (TypedDict state stores strings)
    def _init_rag_state_node(self, state: AssistantState) -> dict:
        """Initialise, reset state before next iteration"""
        return {
            "query_type": "undefined",
            "message_for_user": "",
            "reference_material": {},
        }

    def _classify_query_node(self, state: AssistantState) -> dict:
        """LLM decides what type the user query is"""
        assert self.model
        self.logger.debug(f"{state =}")

        # -- NML_Assistant: state.messages / state.query (BaseModel attrs)
        # -- Here: state.get() / state["key"] (TypedDict is a dict)
        messages = state.get("messages", [])
        messages.append(HumanMessage(content=state["query"]))

        system_prompt = dedent("""
            You are an expert query classifier for a C. elegans research assistant.
            Classify the user input into exactly one category based on its intent.

            Valid categories (in order of priority):

            - question: The query is a request for information — either general
              C. elegans biology/neuroscience, OR about a specific neuron or
              anatomical structure (e.g. AWB, AVAL, pharynx). Neuron and
              anatomy queries should use the research corpus, not WormBase
              (whose neuron endpoints are unreliable).
            - task: The user is asking you to perform an action OR is asking
              about a SPECIFIC gene, protein, or phenotype. This includes:
              * Run a simulation (Hodgkin-Huxley, etc.)
              * Look up or query a gene or protein by name (e.g. eat-4,
                unc-17, cat-4, daf-2) or WBGene ID
              * Look up a phenotype by name or WBPhenotype ID
              * Search a database (WormBase, etc.)
              * Any computational action

            IMPORTANT: If the query mentions a specific gene or protein name,
            classify it as "task" — these should be looked up on WormBase.
            But queries about specific NEURONS or ANATOMY (e.g. AWB, AVAL,
            pharyngeal neurons) should be classified as "question" so they
            are answered from the research corpus.

            Rules:

            - Choose exactly ONE category
            - Base your decision on semantic intent
            - Do not explain your reasoning
            - Do not include any other additional text
            - Provide your answer ONLY as a JSON object matching the requested schema.
            - Take past conversation history and context into account.

            Examples:

            - "How many neurons does C. elegans have?": {{"query_type": "question"}}
            - "What genes are involved in locomotion?": {{"query_type": "question"}}
            - "Describe the pharyngeal nervous system": {{"query_type": "question"}}
            - "What is the capital of France?": {{"query_type": "question"}}
            - "What are we talking about?": {{"query_type": "question"}}
            - "Tell me about C. elegans development": {{"query_type": "question"}}
            - "Tell me about AWB interneurons": {{"query_type": "question"}}
            - "What does AVAL do?": {{"query_type": "question"}}
            - "Describe the ASE sensory neurons": {{"query_type": "question"}}
            - "Run an HH simulation with 10nA current": {{"query_type": "task"}}
            - "Look up eat-4 on WormBase": {{"query_type": "task"}}
            - "What does unc-17 encode?": {{"query_type": "task"}}
            - "What neurotransmitter does eat-4 transport?": {{"query_type": "task"}}
            - "Tell me about the cat-4 gene": {{"query_type": "task"}}
            - "What phenotypes are associated with unc-13?": {{"query_type": "task"}}
            - "What happens when we vary current injection?": {{"query_type": "task"}}
            - "Search WormBase for dopamine receptors": {{"query_type": "task"}}
            - "What is daf-2?": {{"query_type": "task"}}
            - "What are the functions of mec-4?": {{"query_type": "task"}}
            """)

        system_prompt += add_memory_to_prompt(
            messages=state.get("messages", []),
            context_summary=state.get("context_summary", ""),
            num_recent_messages=self.num_recent_messages,
        )

        prompt_template = ChatPromptTemplate(
            [("system", system_prompt), ("human", "User query: {query}")]
        )

        query_node_llm = self.model.with_structured_output(
            QueryTypeSchema, method="json_schema", include_raw=True
        )
        prompt = prompt_template.invoke({"query": state["query"]})

        self.logger.debug(f"{prompt = }")

        output = query_node_llm.invoke(
            prompt, config={"configurable": {"temperature": 0.3}}
        )
        if output["parsing_error"]:
            query_type_result = parse_output_with_thought(
                output["raw"], QueryTypeSchema
            )
        else:
            query_type_result = output["parsed"]
            if isinstance(query_type_result, str):
                query_type_result = QueryTypeSchema(query_type=query_type_result)
            elif isinstance(query_type_result, dict):
                query_type_result = QueryTypeSchema(**query_type_result)
            else:
                if not isinstance(query_type_result, QueryTypeSchema):
                    self.logger.critical(
                        f"Received unexpected query classification: {query_type_result =}"
                    )
                    query_type_result = QueryTypeSchema(query_type="undefined")

        self.logger.debug(f"{query_type_result =}")
        # -- NML_Assistant: returns full QueryTypeSchema object
        # -- Here: extract the string — TypedDict state stores strings
        return {
            "query_type": query_type_result.query_type,
            "messages": messages,
        }

    def _route_query_node(self, state: AssistantState) -> str:
        """Route the query depending on LLM's result"""
        self.logger.debug(f"{state =}")
        # -- NML_Assistant: state.query_type.query_type (nested BaseModel)
        # -- Here: state["query_type"] is already the string
        query_type = state["query_type"]

        return query_type

    async def _tool_node(self, state: AssistantState) -> dict:
        """MCP tool-use node — replaces the code stub in NML_Assistant.

        Uses the LLM to:
        1. Pick the right MCP tool and parameters from the user query
        2. Execute the tool call(s) via FastMCP client
        3. Interpret the results into a natural language answer
        """
        assert self.model
        from fastmcp import Client

        # Fetch available tools from the MCP server
        try:
            async with Client(self.mcp_server) as client:
                mcp_tools = await client.list_tools()
        except Exception as e:
            self.logger.error(f"Could not connect to MCP server: {e}")
            return {
                "message_for_user": (
                    f"Could not initialise tool server: {e}. "
                    "Check that openworm_mcp is installed correctly."
                )
            }

        # Build tool descriptions for the LLM — keep concise to stay
        # within HF token limits
        tool_descriptions = []
        for t in mcp_tools:
            # Only include the properties dict, not the full schema
            props = t.inputSchema.get("properties", {}) if t.inputSchema else {}
            params_brief = {k: v.get("type", "?") for k, v in props.items()}
            # Truncate description to first 200 chars
            desc = t.description[:200] if t.description else ""
            tool_descriptions.append(
                f"- **{t.name}**: {desc}\n  Parameters: {json.dumps(params_brief)}"
            )
        tools_text = "\n\n".join(tool_descriptions)

        # Use raw messages to avoid LangChain escaping JSON braces in
        # tool descriptions as template variables
        plan_messages = [
            SystemMessage(
                content=(
                    "You are a C. elegans research assistant with access to computational tools.\n"
                    "Use the tools below to answer the user's request.\n\n"
                    "# Available tools:\n\n"
                    f"{tools_text}\n\n"
                    "# Instructions:\n\n"
                    "- Pick the most appropriate tool for the user's request.\n"
                    "- ONLY include parameters that the user explicitly mentioned.\n"
                    "  Omit all other parameters — the tool has correct domain-specific\n"
                    "  defaults that should not be overridden.\n"
                    "- If the user asks an open-ended question (e.g. 'what happens when\n"
                    "  we vary X'), plan a set of calls to explore the parameter space.\n"
                    '- Respond ONLY with a JSON object: {"tool": "<name>", "params": {...}, "reasoning": "..."}\n'
                    "- For MULTIPLE calls, respond with a JSON array of such objects.\n"
                    "- If no tool is appropriate, say so in plain text.\n"
                )
            ),
            HumanMessage(content=f"User request: {state['query']}"),
        ]

        # Step 1: LLM decides which tool(s) to call
        plan_response = self.model.invoke(
            plan_messages, config={"configurable": {"temperature": 0.3}}
        )
        plan_text = plan_response.content
        self.logger.debug(f"Tool plan: {plan_text}")

        # Parse the tool call plan
        try:
            cleaned = plan_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
                cleaned = cleaned.rsplit("```", 1)[0]
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                calls = [parsed]
            elif isinstance(parsed, list):
                # Handle case where LLM returns ["json string"] instead of [{...}]
                calls = []
                for item in parsed:
                    if isinstance(item, dict):
                        calls.append(item)
                    elif isinstance(item, str):
                        calls.append(json.loads(item))
                    else:
                        self.logger.warning(f"Unexpected item in plan: {item}")
            else:
                calls = [{"tool": "", "params": {}, "reasoning": str(parsed)}]
        except json.JSONDecodeError:
            # LLM didn't return valid JSON — return its response directly
            self.logger.warning(f"Could not parse tool plan as JSON: {plan_text}")
            return {"message_for_user": plan_text}

        # Step 2: Execute each tool call
        results = []
        async with Client(self.mcp_server) as client:
            for call in calls:
                tool_name = call.get("tool", "")
                params = call.get("params", {})
                reasoning = call.get("reasoning", "")
                self.logger.info(f"Calling tool: {tool_name} with params: {params}")
                try:
                    result = await client.call_tool(tool_name, params)
                    result_texts = []
                    for block in result.content:
                        result_texts.append(
                            block.text if hasattr(block, "text") else str(block)
                        )
                    results.append(
                        {
                            "tool": tool_name,
                            "params": params,
                            "reasoning": reasoning,
                            "output": "\n".join(result_texts),
                        }
                    )
                except Exception as e:
                    self.logger.error(f"Tool call failed: {e}")
                    results.append(
                        {
                            "tool": tool_name,
                            "params": params,
                            "reasoning": reasoning,
                            "error": str(e),
                        }
                    )

        # Extract plot data before stripping from LLM context.
        # We capture both the base64 matplotlib image (if sandbox generated
        # one) and the raw voltage trace arrays for client-side rendering.
        extracted_plot_base64 = ""
        extracted_plot_data = {}
        for r in results:
            if "output" not in r:
                continue
            try:
                _parsed = json.loads(r["output"])
                # Handle stdout as JSON string or dict
                _stdout_raw = (
                    _parsed.get("stdout", "{}") if isinstance(_parsed, dict) else "{}"
                )
                if isinstance(_stdout_raw, str):
                    _stdout = json.loads(_stdout_raw)
                elif isinstance(_stdout_raw, dict):
                    _stdout = _stdout_raw
                else:
                    continue

                if isinstance(_stdout, dict):
                    pb64 = _stdout.get("plot_base64", "")
                    if pb64:
                        extracted_plot_base64 = pb64
                    trace = _stdout.get("voltage_trace", {})
                    if isinstance(trace, dict):
                        t_ms = trace.get("t_ms", [])
                        v_mv = trace.get("v_mv", [])
                        if t_ms and v_mv:
                            extracted_plot_data = {"t_ms": t_ms, "v_mv": v_mv}
                    if extracted_plot_base64 or extracted_plot_data:
                        break

                # Fallback: output IS the stdout directly (no wrapper)
                if not extracted_plot_data and isinstance(_parsed, dict):
                    trace = _parsed.get("voltage_trace", {})
                    if (
                        isinstance(trace, dict)
                        and trace.get("t_ms")
                        and trace.get("v_mv")
                    ):
                        extracted_plot_data = {
                            "t_ms": trace["t_ms"],
                            "v_mv": trace["v_mv"],
                        }
                        break

            except (json.JSONDecodeError, AttributeError, TypeError):
                pass

        # Strip large binary fields (base64 images, full traces) before
        # sending to LLM — they blow up the token count
        for r in results:
            if "output" in r:
                try:
                    parsed = json.loads(r["output"])
                    stdout = (
                        json.loads(parsed.get("stdout", "{}"))
                        if isinstance(parsed.get("stdout"), str)
                        else parsed.get("stdout", {})
                    )
                    # Remove base64 images and truncate large traces
                    for key in list(stdout.keys()):
                        if "base64" in key or "plot" in key.lower():
                            stdout[key] = "[image omitted]"
                        elif key == "voltage_trace":
                            trace = stdout[key]
                            if isinstance(trace, dict):
                                for tk in trace:
                                    if (
                                        isinstance(trace[tk], list)
                                        and len(trace[tk]) > 20
                                    ):
                                        trace[tk] = (
                                            trace[tk][:10] + ["..."] + trace[tk][-10:]
                                        )
                    parsed["stdout"] = (
                        json.dumps(stdout)
                        if isinstance(parsed.get("stdout"), str)
                        else stdout
                    )
                    r["output"] = json.dumps(parsed)
                except (json.JSONDecodeError, AttributeError, TypeError):
                    pass  # leave non-JSON outputs as-is

        # Step 3: LLM interprets results for the user
        interpret_messages = [
            SystemMessage(
                content=(
                    "You are a C. elegans research assistant. The user asked a question "
                    "and tools were called to answer it. Interpret the tool results into "
                    "a clear, helpful answer.\n\n"
                    "- Use specific numbers from the results.\n"
                    "- If multiple calls were made, compare and summarise.\n"
                    "- Use formal but accessible scientific language.\n"
                    "- If a tool returned an error (e.g. HTTP 500), explain what went wrong\n"
                    "  and suggest the user try rephrasing as a general question (e.g.\n"
                    "  'Tell me about AWB neurons' instead of a database lookup) so the\n"
                    "  answer can come from the research corpus instead.\n"
                    "- IMPORTANT: ONLY report information that appears in the tool results.\n"
                    "  Do NOT fill in missing information from your own knowledge. If the\n"
                    "  tool failed or returned incomplete data, say so — do not guess.\n"
                )
            ),
            HumanMessage(
                content=(
                    f"Original request: {state['query']}\n\n"
                    f"Tool results:\n{json.dumps(results, indent=2)}"
                )
            ),
        ]

        interpretation = self.model.invoke(
            interpret_messages, config={"configurable": {"temperature": 0.3}}
        )

        result = {"message_for_user": interpretation.content}
        if extracted_plot_base64:
            result["plot_base64"] = extracted_plot_base64
        if extracted_plot_data:
            result["plot_data"] = extracted_plot_data
        return result

    def _setup_chat_model(self):
        """Set up the LLM chat model"""
        self.model = setup_llm(self.chat_model, self.logger)

    async def setup(self):
        """Set up basics."""
        self._setup_chat_model()
        await self._create_graph()

    async def _create_graph(self):
        """Create the LangGraph"""
        self.workflow = StateGraph(AssistantState)
        self.workflow.add_node("init_state", self._init_rag_state_node)
        self.workflow.add_node("classify_query", self._classify_query_node)

        self._rag_node = RAG(
            vs_config_file=self.vs_config_file, chat_model=self.chat_model, memory=False
        )
        self._rag_node_graph = await self._rag_node.get_graph()
        self.workflow.add_node("rag_graph", self._rag_node_graph)
        self.workflow.add_node("tool_graph", self._tool_node)

        self.workflow.add_edge(START, "init_state")
        self.workflow.add_edge("init_state", "classify_query")

        self.workflow.add_conditional_edges(
            "classify_query",
            self._route_query_node,
            {
                "undefined": "rag_graph",
                "question": "rag_graph",
                "task": "tool_graph",
            },
        )
        self.workflow.add_edge("tool_graph", END)

        self.graph = self.workflow.compile(checkpointer=self.checkpointer)
        if not os.environ.get("RUNNING_IN_DOCKER", 0):
            try:
                self.graph.get_graph().draw_mermaid_png(
                    output_file_path="openworm-assistant-lang-graph.png"
                )
            except BaseException as e:
                self.logger.error("Something went wrong generating lang graph png")
                self.logger.error(e)

    async def run_graph_invoke_state(
        self, state: dict, thread_id: str = "default_thread"
    ):
        """Run the graph but accept and return states"""

        config = {"configurable": {"thread_id": thread_id}}

        if "query" not in state:
            self.logger.error(f"Provided state should include the key 'query': {state}")
            sys.exit(-1)

        final_state = await self.graph.ainvoke(state, config=config)
        self.logger.debug(final_state)
        return final_state

    async def run_graph_invoke(self, query: str, thread_id: str = "default_thread"):
        """Run the graph by using and returning string input"""

        config = {"configurable": {"thread_id": thread_id}}

        final_state = await self.graph.ainvoke({"query": query}, config=config)

        self.logger.debug(f"{final_state =}")
        if message := final_state.get("message_for_user", None):
            return message
        else:
            return "I was unable to answer"

    async def run_graph_stream(self, query: str, thread_id: str = "default_thread"):
        """Run the graph but return the stream"""
        config = {"configurable": {"thread_id": thread_id}}

        for chunk in self.graph.astream({"query": query}, config=config):
            for node, state in chunk.items():
                self.logger.debug(f"{node}: {repr(state)}")
                if message := state.get("message_for_user", None):
                    self.logger.info(f"User message: {message}")
                    yield message
                else:
                    self.logger.debug(f"Working in node: {node}")

    async def graph_stream(self, query: str, thread_id: str = "default_thread"):
        """Run the graph but return the stream"""
        config = {"configurable": {"thread_id": thread_id}}

        res = await self.graph.astream({"query": query}, config=config)
        return res
