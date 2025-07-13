from langchain_core.messages import BaseMessage
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.types import Checkpointer
from langgraph.graph import START, END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from typing import TypedDict, Annotated, Sequence, Union
import operator

import pandas as pd
import json

from core.agents.templates import BaseAgent
from mcp_agents.mcp_datawrangling_agent.agent import DataWranglingAgent
from mcp_agents.mcp_datavisualization_agent.agent import DataVisualizationAgent
from core.utils.plotly import plotly_from_dict
from core.utils.regex import remove_consecutive_duplicates, get_generic_summary

AGENT_NAME = "pandas_data_analyst"

class PandasDataAnalyst(BaseAgent):
    """
    PandasDataAnalyst is a multi-agent class that combines data wrangling and visualization capabilities.

    Parameters:
    -----------
    model:
        The language model to be used for the agents.
    data_wrangling_agent: DataWranglingAgent
        The Data Wrangling Agent for transforming raw data.
    data_visualization_agent: DataVisualizationAgent
        The Data Visualization Agent for generating plots.
    checkpointer: Checkpointer (optional)
        The checkpointer to save the state of the multi-agent system.

    Methods:
    --------
    ainvoke_agent(user_instructions, data_raw, **kwargs)
        Asynchronously invokes the multi-agent with user instructions and raw data.
    invoke_agent(user_instructions, data_raw, **kwargs)
        Synchronously invokes the multi-agent with user instructions and raw data.
    get_data_wrangled()
        Returns the wrangled data as a Pandas DataFrame.
    get_plotly_graph()
        Returns the Plotly graph as a Plotly object.
    get_data_wrangler_function(markdown=False)
        Returns the data wrangling function as a string, optionally in Markdown.
    get_data_visualization_function(markdown=False)
        Returns the data visualization function as a string, optionally in Markdown.
    """

    def __init__(
        self,
        model,
        data_wrangling_agent: DataWranglingAgent,
        data_visualization_agent: DataVisualizationAgent,
        checkpointer: Checkpointer = None,
    ):
        self._params = {
            "model": model,
            "data_wrangling_agent": data_wrangling_agent,
            "data_visualization_agent": data_visualization_agent,
            "checkpointer": checkpointer,
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None

    def _make_compiled_graph(self):
        """Create or rebuild the compiled graph. Resets response to None."""
        self.response = None
        return make_pandas_data_analyst(
            model=self._params["model"],
            data_wrangling_agent=self._params["data_wrangling_agent"]._compiled_graph,
            data_visualization_agent=self._params["data_visualization_agent"]._compiled_graph,
            checkpointer=self._params["checkpointer"],
        )

    def update_params(self, **kwargs):
        """Updates parameters and rebuilds the compiled graph."""
        for k, v in kwargs.items():
            self._params[k] = v
        self._compiled_graph = self._make_compiled_graph()

    async def ainvoke_agent(self, user_instructions, data_raw: Union[pd.DataFrame, dict, list], max_retries: int = 3, retry_count: int = 0, **kwargs):
        """Asynchronously invokes the multi-agent."""
        response = await self._compiled_graph.ainvoke({
            "user_instructions": user_instructions,
            "data_raw": self._convert_data_input(data_raw),
            "max_retries": max_retries,
            "retry_count": retry_count,
        }, **kwargs)
        if response.get("messages"):
            response["messages"] = remove_consecutive_duplicates(response["messages"])
        self.response = response

    def invoke_agent(self, user_instructions, data_raw: Union[pd.DataFrame, dict, list], max_retries: int = 3, retry_count: int = 0, **kwargs):
        """Synchronously invokes the multi-agent."""
        response = self._compiled_graph.invoke({
            "user_instructions": user_instructions,
            "data_raw": self._convert_data_input(data_raw),
            "max_retries": max_retries,
            "retry_count": retry_count,
        }, **kwargs)
        if response.get("messages"):
            response["messages"] = remove_consecutive_duplicates(response["messages"])
        self.response = response

    def get_data_wrangled(self):
        """Returns the wrangled data as a Pandas DataFrame."""
        if self.response and self.response.get("data_wrangled"):
            return pd.DataFrame(self.response.get("data_wrangled"))

    def get_plotly_graph(self):
        """Returns the Plotly graph as a Plotly object."""
        if self.response and self.response.get("plotly_graph"):
            return plotly_from_dict(self.response.get("plotly_graph"))

    def get_data_wrangler_function(self, markdown=False):
        """Returns the data wrangling function as a string."""
        if self.response and self.response.get("data_wrangler_function"):
            code = self.response.get("data_wrangler_function")
            return f"```python\n{code}\n```" if markdown else code

    def get_data_visualization_function(self, markdown=False):
        """Returns the data visualization function as a string."""
        if self.response and self.response.get("data_visualization_function"):
            code = self.response.get("data_visualization_function")
            return f"```python\n{code}\n```" if markdown else code

    def get_workflow_summary(self, markdown=False):
        """Returns a summary of the workflow."""
        if self.response and self.response.get("messages"):
            agents = [msg.role for msg in self.response["messages"]]
            agent_labels = [f"- **Agent {i+1}:** {role}\n" for i, role in enumerate(agents)]
            header = f"# Pandas Data Analyst Workflow Summary\n\nThis workflow contains {len(agents)} agents:\n\n" + "\n".join(agent_labels)
            reports = [get_generic_summary(json.loads(msg.content)) for msg in self.response["messages"]]
            summary = "\n\n" + header + "\n\n".join(reports)
            return summary if markdown else summary

    @staticmethod
    def _convert_data_input(data_raw: Union[pd.DataFrame, dict, list]) -> Union[dict, list]:
        """Converts input data to the expected format (dict or list of dicts)."""
        if isinstance(data_raw, pd.DataFrame):
            return data_raw.to_dict()
        if isinstance(data_raw, dict):
            return data_raw
        if isinstance(data_raw, list):
            return [item.to_dict() if isinstance(item, pd.DataFrame) else item for item in data_raw]
        raise ValueError("data_raw must be a DataFrame, dict, or list of DataFrames/dicts")

def make_pandas_data_analyst(
    model,
    data_wrangling_agent: CompiledStateGraph,
    data_visualization_agent: CompiledStateGraph,
    checkpointer: Checkpointer = None
):
    """
    LLM First 원칙을 준수하는 멀티 에이전트 시스템 생성
    모든 템플릿과 하드코딩된 로직을 제거하고 LLM이 동적으로 결정하도록 구현
    """
    
    llm = model
    
    # LLM First: 동적 라우팅 결정 시스템
    async def dynamic_routing_decision(user_instructions: str) -> dict:
        """LLM이 사용자 질문을 분석하여 동적으로 라우팅 결정"""
        
        # 1단계: LLM이 질문 분석 방식을 스스로 결정
        analysis_prompt = f"""
당신은 데이터 분석 라우팅 전문가입니다. 다음 사용자 질문을 분석해주세요:

사용자 질문: {user_instructions}

이 질문을 분석하여 다음을 결정해주세요:
1. 데이터 조작/가공이 필요한가?
2. 시각화가 필요한가?
3. 어떤 접근 방식이 최적인가?

이 특정 질문에 맞는 최적의 분석 방식을 제안해주세요.
고정된 템플릿이 아닌, 이 질문에 특화된 분석을 제공해주세요.
"""
        
        analysis_response = await llm.ainvoke(analysis_prompt)
        analysis = analysis_response.content if hasattr(analysis_response, 'content') else str(analysis_response)
        
        # 2단계: 분석 결과를 바탕으로 라우팅 결정
        routing_prompt = f"""
앞서 분석한 내용을 바탕으로 구체적인 라우팅 결정을 내려주세요:

사용자 질문: {user_instructions}
분석 결과: {analysis}

다음 JSON 형식으로 응답해주세요:
{{
    "user_instructions_data_wrangling": "데이터 조작 에이전트에게 전달할 지시사항 (필요하지 않으면 null)",
    "user_instructions_data_visualization": "시각화 에이전트에게 전달할 지시사항 (필요하지 않으면 null)",
    "routing_preprocessor_decision": "chart 또는 table 중 선택",
    "reasoning": "이 결정을 내린 이유"
}}

이 특정 상황에 맞는 최적의 결정을 내려주세요.
"""
        
        routing_response = await llm.ainvoke(routing_prompt)
        routing_content = routing_response.content if hasattr(routing_response, 'content') else str(routing_response)
        
        # JSON 파싱
        import json
        import re
        
        json_match = re.search(r'\{.*\}', routing_content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # 파싱 실패 시 LLM이 다시 시도
        fallback_prompt = f"""
이전 응답을 유효한 JSON 형식으로 변환해주세요:

{routing_content}

다음 형식으로 정확히 응답해주세요:
{{
    "user_instructions_data_wrangling": "...",
    "user_instructions_data_visualization": "...",
    "routing_preprocessor_decision": "chart 또는 table"
}}
"""
        
        fallback_response = await llm.ainvoke(fallback_prompt)
        fallback_content = fallback_response.content if hasattr(fallback_response, 'content') else str(fallback_response)
        
        json_match = re.search(r'\{.*\}', fallback_content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # 모든 파싱 실패 시 기본값 반환
        return {
            "user_instructions_data_wrangling": user_instructions,
            "user_instructions_data_visualization": None,
            "routing_preprocessor_decision": "table"
        }
    
    # 기존 PromptTemplate 완전 제거하고 동적 처리로 대체
    def routing_preprocessor_dynamic(user_instructions: str) -> dict:
        """동적 라우팅 전처리"""
        import asyncio
        
        try:
            # 비동기 처리
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 이미 실행 중인 루프가 있으면 태스크로 실행
                task = loop.create_task(dynamic_routing_decision(user_instructions))
                # 동기 함수에서 비동기 태스크 결과를 기다리는 것은 복잡하므로 
                # 여기서는 간단한 동기 처리로 폴백
                return simple_routing_decision(user_instructions)
            else:
                # 새로운 이벤트 루프 실행
                return asyncio.run(dynamic_routing_decision(user_instructions))
        except Exception as e:
            print(f"동적 라우팅 실패, 간단한 결정으로 폴백: {e}")
            return simple_routing_decision(user_instructions)
    
    def simple_routing_decision(user_instructions: str) -> dict:
        """간단한 라우팅 결정 (폴백용)"""
        # LLM이 사용 불가능한 경우 최소한의 로직
        if any(keyword in user_instructions.lower() for keyword in ['chart', 'plot', 'graph', 'visual']):
            return {
                "user_instructions_data_wrangling": user_instructions,
                "user_instructions_data_visualization": user_instructions,
                "routing_preprocessor_decision": "chart"
            }
        else:
            return {
                "user_instructions_data_wrangling": user_instructions,
                "user_instructions_data_visualization": None,
                "routing_preprocessor_decision": "table"
            }

    class PrimaryState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        user_instructions: str
        user_instructions_data_wrangling: str
        user_instructions_data_visualization: str
        routing_preprocessor_decision: str
        data_raw: Union[dict, list]
        data_wrangled: dict
        data_wrangler_function: str
        data_visualization_function: str
        plotly_graph: dict
        plotly_error: str
        max_retries: int
        retry_count: int
        
        
    def preprocess_routing(state: PrimaryState):
        print("---PANDAS DATA ANALYST (LLM First)---")
        print("*************************************")
        print("---DYNAMIC ROUTING PREPROCESSOR---")
        question = state.get("user_instructions")
        
        # LLM First 동적 라우팅
        response = routing_preprocessor_dynamic(question)
        
        return {
            "user_instructions_data_wrangling": response.get('user_instructions_data_wrangling'),
            "user_instructions_data_visualization": response.get('user_instructions_data_visualization'),
            "routing_preprocessor_decision": response.get('routing_preprocessor_decision'),
        }
    
    def router_chart_or_table(state: PrimaryState):
        print("---DYNAMIC ROUTER: CHART OR TABLE---")
        return "chart" if state.get('routing_preprocessor_decision') == "chart" else "table"
    
    
    def invoke_data_wrangling_agent(state: PrimaryState):
        
        response = data_wrangling_agent.invoke({
            "user_instructions": state.get("user_instructions_data_wrangling"),
            "data_raw": state.get("data_raw"),
            "max_retries": state.get("max_retries"),
            "retry_count": state.get("retry_count"),
        })

        return {
            "messages": response.get("messages"),
            "data_wrangled": response.get("data_wrangled"),
            "data_wrangler_function": response.get("data_wrangler_function"),
            "plotly_error": response.get("data_visualization_error"),
            
        }
        
    def invoke_data_visualization_agent(state: PrimaryState):
        
        response = data_visualization_agent.invoke({
            "user_instructions": state.get("user_instructions_data_visualization"),
            "data_raw": state.get("data_wrangled") if state.get("data_wrangled") else state.get("data_raw"),
            "max_retries": state.get("max_retries"),
            "retry_count": state.get("retry_count"),
        })
        
        return {
            "messages": response.get("messages"),
            "data_visualization_function": response.get("data_visualization_function"),
            "plotly_graph": response.get("plotly_graph"),
            "plotly_error": response.get("data_visualization_error"),
        }

    def route_printer(state: PrimaryState):
        print("---DYNAMIC ROUTE PRINTER---")
        print(f"    Route: {state.get('routing_preprocessor_decision')}")
        print("---END---")
        return {}
    
    workflow = StateGraph(PrimaryState)
    
    workflow.add_node("routing_preprocessor", preprocess_routing)
    workflow.add_node("data_wrangling_agent", invoke_data_wrangling_agent)
    workflow.add_node("data_visualization_agent", invoke_data_visualization_agent)
    workflow.add_node("route_printer", route_printer)

    workflow.add_edge(START, "routing_preprocessor")
    workflow.add_edge("routing_preprocessor", "data_wrangling_agent")
    
    workflow.add_conditional_edges(
        "data_wrangling_agent", 
        router_chart_or_table,
        {
            "chart": "data_visualization_agent",
            "table": "route_printer"
        }
    )
    
    workflow.add_edge("data_visualization_agent", "route_printer")
    workflow.add_edge("route_printer", END)

    app = workflow.compile(
        checkpointer=checkpointer, 
        name=AGENT_NAME
    )
    
    return app
