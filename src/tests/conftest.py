import sys
from pathlib import Path
import types
from langchain_core.messages import ToolMessage


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _install_langgraph_stubs() -> None:
    # Keep tests runnable even when langgraph is not installed in CI/local env.
    if "langgraph" in sys.modules:
        return

    langgraph_mod = types.ModuleType("langgraph")
    types_mod = types.ModuleType("langgraph.types")
    errors_mod = types.ModuleType("langgraph.errors")
    graph_mod = types.ModuleType("langgraph.graph")
    message_mod = types.ModuleType("langgraph.graph.message")
    checkpoint_mod = types.ModuleType("langgraph.checkpoint")
    redis_mod = types.ModuleType("langgraph.checkpoint.redis")

    class GraphInterrupt(Exception):
        pass

    class RedisSaver:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def setup(self):
            return None

    def interrupt(payload):
        return payload

    def add_messages(left, right):
        left = left or []
        right = right or []
        return list(left) + list(right)

    types_mod.interrupt = interrupt
    errors_mod.GraphInterrupt = GraphInterrupt
    message_mod.add_messages = add_messages
    redis_mod.RedisSaver = RedisSaver

    sys.modules["langgraph"] = langgraph_mod
    sys.modules["langgraph.types"] = types_mod
    sys.modules["langgraph.errors"] = errors_mod
    sys.modules["langgraph.graph"] = graph_mod
    sys.modules["langgraph.graph.message"] = message_mod
    sys.modules["langgraph.checkpoint"] = checkpoint_mod
    sys.modules["langgraph.checkpoint.redis"] = redis_mod


_install_langgraph_stubs()


def _install_langchain_groq_stub() -> None:
    if "langchain_groq" in sys.modules:
        return

    groq_mod = types.ModuleType("langchain_groq")

    class ChatGroq:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def invoke(self, *_args, **_kwargs):
            raise RuntimeError("ChatGroq test stub should be monkeypatched in tests")

    groq_mod.ChatGroq = ChatGroq
    sys.modules["langchain_groq"] = groq_mod


_install_langchain_groq_stub()


def _install_external_service_stubs() -> None:
    if "groq" not in sys.modules:
        groq_mod = types.ModuleType("groq")

        class Groq:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        groq_mod.Groq = Groq
        sys.modules["groq"] = groq_mod

    if "google" not in sys.modules:
        google_mod = types.ModuleType("google")
        cloud_mod = types.ModuleType("google.cloud")
        vision_mod = types.ModuleType("google.cloud.vision")
        oauth2_mod = types.ModuleType("google.oauth2")
        sa_mod = types.ModuleType("google.oauth2.service_account")

        class ImageAnnotatorClient:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        class Credentials:
            @staticmethod
            def from_service_account_file(*args, **kwargs):
                return object()

            @staticmethod
            def from_service_account_info(*args, **kwargs):
                return object()

        vision_mod.ImageAnnotatorClient = ImageAnnotatorClient
        sa_mod.Credentials = Credentials
        sys.modules["google"] = google_mod
        sys.modules["google.cloud"] = cloud_mod
        sys.modules["google.cloud.vision"] = vision_mod
        sys.modules["google.oauth2"] = oauth2_mod
        sys.modules["google.oauth2.service_account"] = sa_mod


_install_external_service_stubs()


def _install_backend_node_stubs() -> None:
    # Avoid importing heavy backend.agents.nodes package during unit tests.
    if "backend.agents.nodes" in sys.modules:
        return

    nodes_mod = types.ModuleType("backend.agents.nodes")
    nodes_mod.__path__ = []

    class BaseAgent:
        pass

    nodes_mod.BaseAgent = BaseAgent
    nodes_mod.AgentState = dict
    nodes_mod.payload = lambda *args, **kwargs: None
    nodes_mod.render_md = lambda *args, **kwargs: ""
    nodes_mod._HARD_BLOCK_KEYWORDS = [
        "synthesise",
        "synthesize",
        "explosive",
        "poison",
    ]
    nodes_mod.__all__ = [
        "BaseAgent",
        "AgentState",
        "payload",
        "render_md",
        "_HARD_BLOCK_KEYWORDS",
        "IntentRouterOutput",
        "calculator_tool",
        "web_search_tool",
        "rag_tool",
        "ToolMessage",
    ]

    memory_mod = types.ModuleType("backend.agents.nodes.memory")
    memory_mod.REDIS_URL = "redis://localhost:6379/0"
    memory_mod.TIKTOKEN_MODEL = "gpt-4o"
    memory_mod.STM_SUMMARY_TTL = 7200
    memory_mod.THREAD_TTL = 86400
    memory_mod.MAX_THREADS_SHOWN = 20
    memory_mod.TOKEN_LIMIT = 8000
    memory_mod.KEEP_LAST_N = 6
    memory_mod.EPISODIC_TTL = 7776000
    memory_mod.DECAY_THRESHOLD = 0.05
    memory_mod.TOP_K_EPISODES = 3
    memory_mod.EPISODIC_INDEX_NAME = "idx:episodic"
    memory_mod.EPISODIC_INDEX_SCHEMA = {"index": {"name": "idx:episodic"}}

    mm_mod = types.ModuleType("backend.agents.nodes.memory.memory_manager")
    mm_mod.trim_messages_if_needed = lambda messages, thread_id, llm: messages
    mm_mod.format_ltm_for_solver = lambda *_args, **_kwargs: ""
    mm_mod.format_ltm_for_explainer = lambda *_args, **_kwargs: ""

    tools_pkg = types.ModuleType("backend.agents.nodes.tools")
    tools_pkg.__path__ = []
    tools_pkg.EMBED_INPUT_TYPE_DOC = "search_document"
    tools_pkg.EMBED_INPUT_TYPE_QUERY = "search_query"
    mcp_pkg = types.ModuleType("backend.agents.nodes.tools.mcp")
    mcp_pkg.TAVILY_REMOTE_MCP_BASE = "https://mcp.tavily.com/mcp/"
    mcp_pkg._TOOL_SEARCH = "tavily_search"
    tools_mod = types.ModuleType("backend.agents.nodes.tools.tools")

    class _WebSearchTool:
        @staticmethod
        def func(query):
            return f"stub:{query}"

    class _CalculatorTool:
        @staticmethod
        def invoke(payload):
            return f"calc:{payload}"

    class _RagTool:
        @staticmethod
        def invoke(payload):
            return f"rag:{payload}"

    tools_mod.web_search_tool = _WebSearchTool()
    tools_mod.calculator_tool = _CalculatorTool()
    tools_mod.rag_tool = _RagTool()
    tools_mod.has_store = lambda _thread_id: False
    tools_mod._embed_texts = lambda texts, _input_type: texts

    nodes_mod.calculator_tool = tools_mod.calculator_tool
    nodes_mod.web_search_tool = tools_mod.web_search_tool
    nodes_mod.rag_tool = tools_mod.rag_tool
    nodes_mod.IntentRouterOutput = dict
    nodes_mod.ToolMessage = ToolMessage

    sys.modules["backend.agents.nodes"] = nodes_mod
    sys.modules["backend.agents.nodes.memory"] = memory_mod
    sys.modules["backend.agents.nodes.memory.memory_manager"] = mm_mod
    sys.modules["backend.agents.nodes.tools"] = tools_pkg
    sys.modules["backend.agents.nodes.tools.mcp"] = mcp_pkg
    sys.modules["backend.agents.nodes.tools.tools"] = tools_mod

    if "redisvl" not in sys.modules:
        redisvl_mod = types.ModuleType("redisvl")
        redisvl_index_mod = types.ModuleType("redisvl.index")
        redisvl_query_mod = types.ModuleType("redisvl.query")

        class SearchIndex:
            @classmethod
            def from_dict(cls, _schema):
                return cls()

            def connect(self, redis_url=None):
                return None

            def create(self, overwrite=False):
                return None

            def query(self, _query):
                return []

        class VectorQuery:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        redisvl_index_mod.SearchIndex = SearchIndex
        redisvl_query_mod.VectorQuery = VectorQuery
        sys.modules["redisvl"] = redisvl_mod
        sys.modules["redisvl.index"] = redisvl_index_mod
        sys.modules["redisvl.query"] = redisvl_query_mod


_install_backend_node_stubs()
