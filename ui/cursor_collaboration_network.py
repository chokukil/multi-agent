"""
Cursor-Style Agent Collaboration Network Visualization
D3.js 기반 A2A Message Router 시각화, 실시간 데이터 흐름, Cursor 스타일 노드 디자인
"""

import streamlit as st
import json
import time
import uuid
import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import streamlit.components.v1 as components

# 로컬 모듈 임포트
from ui.cursor_sse_realtime import get_cursor_sse_realtime, SSEEventType
from ui.cursor_theme_system import get_cursor_theme

class NodeType(Enum):
    """노드 타입"""
    AGENT = "agent"
    MCP_TOOL = "mcp_tool"
    DATA_SOURCE = "data_source"
    OUTPUT = "output"
    ROUTER = "router"
    ORCHESTRATOR = "orchestrator"

class ConnectionType(Enum):
    """연결 타입"""
    A2A_MESSAGE = "a2a_message"
    DATA_FLOW = "data_flow"
    DEPENDENCY = "dependency"
    FEEDBACK = "feedback"
    CONTROL = "control"

class NodeStatus(Enum):
    """노드 상태"""
    IDLE = "idle"
    THINKING = "thinking"
    WORKING = "working"
    COMPLETED = "completed"
    ERROR = "error"
    OFFLINE = "offline"

@dataclass
class NetworkNode:
    """네트워크 노드"""
    id: str
    name: str
    type: NodeType
    status: NodeStatus = NodeStatus.IDLE
    position: Tuple[float, float] = (0, 0)
    metadata: Dict[str, Any] = field(default_factory=dict)
    connections: List[str] = field(default_factory=list)
    message_count: int = 0
    last_activity: float = field(default_factory=time.time)
    processing_time: float = 0.0
    success_rate: float = 1.0

@dataclass
class NetworkConnection:
    """네트워크 연결"""
    id: str
    source: str
    target: str
    type: ConnectionType
    strength: float = 1.0
    active: bool = False
    message_data: Optional[Dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    message_count: int = 0

@dataclass
class MessageFlow:
    """메시지 흐름"""
    id: str
    source: str
    target: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    status: str = "pending"
    route: List[str] = field(default_factory=list)

class CursorCollaborationNetwork:
    """Cursor 스타일 협업 네트워크"""
    
    def __init__(self):
        self.nodes: Dict[str, NetworkNode] = {}
        self.connections: Dict[str, NetworkConnection] = {}
        self.message_flows: List[MessageFlow] = []
        self.network_config = self._get_default_config()
        self.realtime_manager = get_cursor_sse_realtime()
        
        # 기본 네트워크 생성
        self._create_default_network()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """기본 네트워크 설정"""
        return {
            "width": 800,
            "height": 600,
            "node_radius": {
                NodeType.AGENT: 25,
                NodeType.MCP_TOOL: 20,
                NodeType.DATA_SOURCE: 18,
                NodeType.OUTPUT: 15,
                NodeType.ROUTER: 30,
                NodeType.ORCHESTRATOR: 35
            },
            "colors": {
                NodeType.AGENT: "#007acc",
                NodeType.MCP_TOOL: "#2e7d32",
                NodeType.DATA_SOURCE: "#f57c00",
                NodeType.OUTPUT: "#7b1fa2",
                NodeType.ROUTER: "#d32f2f",
                NodeType.ORCHESTRATOR: "#1976d2"
            },
            "status_colors": {
                NodeStatus.IDLE: "#666666",
                NodeStatus.THINKING: "#1a4f8a",
                NodeStatus.WORKING: "#2e7d32",
                NodeStatus.COMPLETED: "#388e3c",
                NodeStatus.ERROR: "#d32f2f",
                NodeStatus.OFFLINE: "#424242"
            },
            "connection_colors": {
                ConnectionType.A2A_MESSAGE: "#007acc",
                ConnectionType.DATA_FLOW: "#4caf50",
                ConnectionType.DEPENDENCY: "#ff9800",
                ConnectionType.FEEDBACK: "#9c27b0",
                ConnectionType.CONTROL: "#f44336"
            },
            "force_settings": {
                "charge": -800,
                "link_distance": 100,
                "center_strength": 0.1,
                "collision_radius": 30
            }
        }
    
    def _create_default_network(self):
        """기본 네트워크 생성"""
        # 오케스트레이터 (중앙 허브)
        self.add_node(
            "orchestrator",
            "A2A Orchestrator",
            NodeType.ORCHESTRATOR,
            metadata={
                "description": "A2A 메시지 라우팅 및 오케스트레이션",
                "capabilities": ["message_routing", "task_coordination", "state_management"]
            }
        )
        
        # 에이전트들
        agents = [
            ("pandas_agent", "Pandas Agent", "데이터 분석 및 처리"),
            ("viz_agent", "Visualization Agent", "데이터 시각화"),
            ("ml_agent", "ML Agent", "머신러닝 모델링"),
            ("knowledge_agent", "Knowledge Agent", "지식 베이스 관리")
        ]
        
        for agent_id, agent_name, description in agents:
            self.add_node(
                agent_id,
                agent_name,
                NodeType.AGENT,
                metadata={
                    "description": description,
                    "capabilities": ["data_processing", "analysis", "reporting"]
                }
            )
            
            # 오케스트레이터와 연결
            self.add_connection(
                f"orchestrator_{agent_id}",
                "orchestrator",
                agent_id,
                ConnectionType.A2A_MESSAGE
            )
        
        # MCP 도구들
        mcp_tools = [
            ("data_loader", "Data Loader", "데이터 로드"),
            ("data_cleaner", "Data Cleaner", "데이터 정제"),
            ("eda_tools", "EDA Tools", "탐색적 데이터 분석"),
            ("viz_tools", "Visualization Tools", "시각화 도구"),
            ("ml_tools", "ML Tools", "머신러닝 도구")
        ]
        
        for tool_id, tool_name, description in mcp_tools:
            self.add_node(
                tool_id,
                tool_name,
                NodeType.MCP_TOOL,
                metadata={
                    "description": description,
                    "tool_type": "mcp"
                }
            )
        
        # 데이터 소스들
        data_sources = [
            ("csv_data", "CSV Data", "CSV 파일 데이터"),
            ("api_data", "API Data", "외부 API 데이터"),
            ("db_data", "Database", "데이터베이스")
        ]
        
        for source_id, source_name, description in data_sources:
            self.add_node(
                source_id,
                source_name,
                NodeType.DATA_SOURCE,
                metadata={
                    "description": description,
                    "data_type": "input"
                }
            )
        
        # 출력 노드들
        outputs = [
            ("dashboard", "Dashboard", "대시보드 출력"),
            ("report", "Report", "보고서 생성"),
            ("model", "Model", "훈련된 모델")
        ]
        
        for output_id, output_name, description in outputs:
            self.add_node(
                output_id,
                output_name,
                NodeType.OUTPUT,
                metadata={
                    "description": description,
                    "output_type": "result"
                }
            )
        
        # 에이전트-도구 연결
        self.add_connection("pandas_data_loader", "pandas_agent", "data_loader", ConnectionType.DATA_FLOW)
        self.add_connection("pandas_data_cleaner", "pandas_agent", "data_cleaner", ConnectionType.DATA_FLOW)
        self.add_connection("viz_viz_tools", "viz_agent", "viz_tools", ConnectionType.DATA_FLOW)
        self.add_connection("ml_ml_tools", "ml_agent", "ml_tools", ConnectionType.DATA_FLOW)
        
        # 데이터 소스-도구 연결
        self.add_connection("csv_loader", "csv_data", "data_loader", ConnectionType.DATA_FLOW)
        self.add_connection("api_loader", "api_data", "data_loader", ConnectionType.DATA_FLOW)
        self.add_connection("db_loader", "db_data", "data_loader", ConnectionType.DATA_FLOW)
        
        # 에이전트-출력 연결
        self.add_connection("viz_dashboard", "viz_agent", "dashboard", ConnectionType.DATA_FLOW)
        self.add_connection("pandas_report", "pandas_agent", "report", ConnectionType.DATA_FLOW)
        self.add_connection("ml_model", "ml_agent", "model", ConnectionType.DATA_FLOW)
        
        # 에이전트 간 협업 연결
        self.add_connection("pandas_viz", "pandas_agent", "viz_agent", ConnectionType.A2A_MESSAGE)
        self.add_connection("pandas_ml", "pandas_agent", "ml_agent", ConnectionType.A2A_MESSAGE)
        self.add_connection("ml_viz", "ml_agent", "viz_agent", ConnectionType.FEEDBACK)
    
    def add_node(self, node_id: str, name: str, node_type: NodeType, 
                 status: NodeStatus = NodeStatus.IDLE, 
                 metadata: Dict[str, Any] = None):
        """노드 추가"""
        node = NetworkNode(
            id=node_id,
            name=name,
            type=node_type,
            status=status,
            metadata=metadata or {}
        )
        self.nodes[node_id] = node
        return node
    
    def add_connection(self, connection_id: str, source: str, target: str, 
                      connection_type: ConnectionType, strength: float = 1.0):
        """연결 추가"""
        connection = NetworkConnection(
            id=connection_id,
            source=source,
            target=target,
            type=connection_type,
            strength=strength
        )
        self.connections[connection_id] = connection
        
        # 노드의 연결 목록 업데이트
        if source in self.nodes:
            self.nodes[source].connections.append(connection_id)
        if target in self.nodes:
            self.nodes[target].connections.append(connection_id)
        
        return connection
    
    def update_node_status(self, node_id: str, status: NodeStatus):
        """노드 상태 업데이트"""
        if node_id in self.nodes:
            self.nodes[node_id].status = status
            self.nodes[node_id].last_activity = time.time()
    
    def activate_connection(self, connection_id: str, message_data: Dict[str, Any] = None):
        """연결 활성화"""
        if connection_id in self.connections:
            connection = self.connections[connection_id]
            connection.active = True
            connection.message_data = message_data
            connection.last_used = time.time()
            connection.message_count += 1
    
    def send_message(self, source: str, target: str, message_type: str, 
                    payload: Dict[str, Any]) -> str:
        """메시지 전송"""
        message_id = str(uuid.uuid4())
        
        # 메시지 흐름 생성
        message_flow = MessageFlow(
            id=message_id,
            source=source,
            target=target,
            message_type=message_type,
            payload=payload,
            route=[source, target]
        )
        
        self.message_flows.append(message_flow)
        
        # 노드 상태 업데이트
        self.update_node_status(source, NodeStatus.WORKING)
        self.update_node_status(target, NodeStatus.THINKING)
        
        # 연결 활성화
        for connection in self.connections.values():
            if connection.source == source and connection.target == target:
                self.activate_connection(connection.id, payload)
        
        return message_id
    
    def get_network_data(self) -> Dict[str, Any]:
        """네트워크 데이터 반환"""
        return {
            "nodes": [asdict(node) for node in self.nodes.values()],
            "connections": [asdict(connection) for connection in self.connections.values()],
            "message_flows": [asdict(flow) for flow in self.message_flows[-10:]],  # 최근 10개
            "config": self.network_config,
            "timestamp": time.time()
        }
    
    def get_d3_visualization(self) -> str:
        """D3.js 시각화 HTML 생성"""
        network_data = self.get_network_data()
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Cursor Collaboration Network</title>
            <script src="https://d3js.org/d3.v7.min.js"></script>
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    background: #1a1a1a;
                    color: #ffffff;
                    font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
                }}
                
                #network {{
                    width: 100%;
                    height: 600px;
                    background: #1a1a1a;
                    border: 1px solid #333;
                    border-radius: 8px;
                }}
                
                .node {{
                    cursor: pointer;
                    transition: all 0.3s ease;
                }}
                
                .node:hover {{
                    stroke-width: 3px;
                    filter: drop-shadow(0 0 10px rgba(0, 122, 204, 0.6));
                }}
                
                .node-label {{
                    font-size: 12px;
                    font-weight: 500;
                    text-anchor: middle;
                    fill: #ffffff;
                    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
                }}
                
                .connection {{
                    stroke-width: 2px;
                    opacity: 0.6;
                    transition: all 0.3s ease;
                }}
                
                .connection.active {{
                    stroke-width: 4px;
                    opacity: 1;
                    animation: pulse 2s ease-in-out infinite;
                }}
                
                .connection:hover {{
                    stroke-width: 3px;
                    opacity: 1;
                }}
                
                @keyframes pulse {{
                    0%, 100% {{ opacity: 0.6; }}
                    50% {{ opacity: 1; }}
                }}
                
                .message-flow {{
                    fill: #007acc;
                    stroke: #ffffff;
                    stroke-width: 2px;
                    r: 4px;
                    opacity: 0.8;
                }}
                
                .tooltip {{
                    position: absolute;
                    background: rgba(45, 45, 45, 0.95);
                    color: #ffffff;
                    padding: 10px;
                    border-radius: 6px;
                    font-size: 12px;
                    pointer-events: none;
                    z-index: 1000;
                    border: 1px solid #555;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
                }}
                
                .status-indicator {{
                    position: absolute;
                    top: 10px;
                    right: 10px;
                    background: rgba(45, 45, 45, 0.9);
                    padding: 10px;
                    border-radius: 6px;
                    font-size: 12px;
                    border: 1px solid #555;
                }}
                
                .legend {{
                    position: absolute;
                    bottom: 10px;
                    left: 10px;
                    background: rgba(45, 45, 45, 0.9);
                    padding: 10px;
                    border-radius: 6px;
                    font-size: 11px;
                    border: 1px solid #555;
                }}
                
                .legend-item {{
                    display: flex;
                    align-items: center;
                    margin: 3px 0;
                }}
                
                .legend-color {{
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    margin-right: 8px;
                }}
            </style>
        </head>
        <body>
            <div id="network"></div>
            <div id="tooltip" class="tooltip" style="display: none;"></div>
            <div class="status-indicator" id="status">
                <div>활성 노드: <span id="active-nodes">0</span></div>
                <div>활성 연결: <span id="active-connections">0</span></div>
                <div>메시지 흐름: <span id="message-flows">0</span></div>
            </div>
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background: #007acc;"></div>
                    <span>Agent</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #2e7d32;"></div>
                    <span>MCP Tool</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #f57c00;"></div>
                    <span>Data Source</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #7b1fa2;"></div>
                    <span>Output</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #d32f2f;"></div>
                    <span>Router</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #1976d2;"></div>
                    <span>Orchestrator</span>
                </div>
            </div>
            
            <script>
                const networkData = {json.dumps(network_data, indent=2)};
                
                // 차트 설정
                const width = 800;
                const height = 600;
                const config = networkData.config;
                
                // SVG 생성
                const svg = d3.select("#network")
                    .append("svg")
                    .attr("width", width)
                    .attr("height", height);
                
                // 그래프 그룹
                const graph = svg.append("g");
                
                // 줌 설정
                const zoom = d3.zoom()
                    .scaleExtent([0.5, 3])
                    .on("zoom", (event) => {{
                        graph.attr("transform", event.transform);
                    }});
                
                svg.call(zoom);
                
                // 포스 시뮬레이션
                const simulation = d3.forceSimulation(networkData.nodes)
                    .force("link", d3.forceLink(networkData.connections).id(d => d.id)
                        .distance(d => config.force_settings.link_distance))
                    .force("charge", d3.forceManyBody()
                        .strength(config.force_settings.charge))
                    .force("center", d3.forceCenter(width / 2, height / 2)
                        .strength(config.force_settings.center_strength))
                    .force("collision", d3.forceCollide()
                        .radius(d => config.node_radius[d.type] + config.force_settings.collision_radius));
                
                // 연결선 생성
                const links = graph.selectAll(".connection")
                    .data(networkData.connections)
                    .enter()
                    .append("line")
                    .attr("class", "connection")
                    .attr("stroke", d => config.connection_colors[d.type])
                    .attr("stroke-width", d => d.active ? 4 : 2)
                    .attr("opacity", d => d.active ? 1 : 0.6);
                
                // 노드 생성
                const nodes = graph.selectAll(".node")
                    .data(networkData.nodes)
                    .enter()
                    .append("g")
                    .attr("class", "node")
                    .call(d3.drag()
                        .on("start", dragStarted)
                        .on("drag", dragged)
                        .on("end", dragEnded));
                
                // 노드 원 추가
                nodes.append("circle")
                    .attr("r", d => config.node_radius[d.type])
                    .attr("fill", d => config.colors[d.type])
                    .attr("stroke", d => config.status_colors[d.status])
                    .attr("stroke-width", 2);
                
                // 노드 라벨 추가
                nodes.append("text")
                    .attr("class", "node-label")
                    .attr("dy", d => config.node_radius[d.type] + 16)
                    .text(d => d.name);
                
                // 상태 표시 점 추가
                nodes.append("circle")
                    .attr("r", 4)
                    .attr("cx", d => config.node_radius[d.type] - 8)
                    .attr("cy", d => -(config.node_radius[d.type] - 8))
                    .attr("fill", d => config.status_colors[d.status]);
                
                // 툴팁 설정
                const tooltip = d3.select("#tooltip");
                
                nodes.on("mouseover", (event, d) => {{
                    tooltip.style("display", "block")
                        .html(`
                            <strong>${{d.name}}</strong><br>
                            타입: ${{d.type}}<br>
                            상태: ${{d.status}}<br>
                            메시지 수: ${{d.message_count}}<br>
                            성공률: ${{(d.success_rate * 100).toFixed(1)}}%<br>
                            ${{d.metadata.description || ''}}
                        `)
                        .style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY - 10) + "px");
                }})
                .on("mouseout", () => {{
                    tooltip.style("display", "none");
                }});
                
                // 연결선 툴팁
                links.on("mouseover", (event, d) => {{
                    tooltip.style("display", "block")
                        .html(`
                            <strong>${{d.source.name}} → ${{d.target.name}}</strong><br>
                            타입: ${{d.type}}<br>
                            강도: ${{d.strength}}<br>
                            메시지 수: ${{d.message_count}}<br>
                            마지막 사용: ${{new Date(d.last_used * 1000).toLocaleTimeString()}}
                        `)
                        .style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY - 10) + "px");
                }})
                .on("mouseout", () => {{
                    tooltip.style("display", "none");
                }});
                
                // 메시지 흐름 애니메이션
                function animateMessageFlow(flow) {{
                    const source = networkData.nodes.find(n => n.id === flow.source);
                    const target = networkData.nodes.find(n => n.id === flow.target);
                    
                    if (source && target) {{
                        const message = graph.append("circle")
                            .attr("class", "message-flow")
                            .attr("r", 6)
                            .attr("cx", source.x)
                            .attr("cy", source.y);
                        
                        message.transition()
                            .duration(2000)
                            .attr("cx", target.x)
                            .attr("cy", target.y)
                            .on("end", () => {{
                                message.remove();
                            }});
                    }}
                }}
                
                // 시뮬레이션 틱
                simulation.on("tick", () => {{
                    links
                        .attr("x1", d => d.source.x)
                        .attr("y1", d => d.source.y)
                        .attr("x2", d => d.target.x)
                        .attr("y2", d => d.target.y);
                    
                    nodes
                        .attr("transform", d => `translate(${{d.x}},${{d.y}})`);
                }});
                
                // 드래그 함수
                function dragStarted(event, d) {{
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    d.fx = d.x;
                    d.fy = d.y;
                }}
                
                function dragged(event, d) {{
                    d.fx = event.x;
                    d.fy = event.y;
                }}
                
                function dragEnded(event, d) {{
                    if (!event.active) simulation.alphaTarget(0);
                    d.fx = null;
                    d.fy = null;
                }}
                
                // 상태 업데이트
                function updateStatus() {{
                    const activeNodes = networkData.nodes.filter(n => n.status !== 'idle' && n.status !== 'offline').length;
                    const activeConnections = networkData.connections.filter(c => c.active).length;
                    const messageFlows = networkData.message_flows.length;
                    
                    document.getElementById('active-nodes').textContent = activeNodes;
                    document.getElementById('active-connections').textContent = activeConnections;
                    document.getElementById('message-flows').textContent = messageFlows;
                }}
                
                // 초기 상태 업데이트
                updateStatus();
                
                // 주기적 업데이트 (실제로는 A2A SDK SSE에서 받아야 함)
                setInterval(() => {{
                    // 랜덤 노드 상태 변경
                    const randomNode = networkData.nodes[Math.floor(Math.random() * networkData.nodes.length)];
                    const statuses = ['idle', 'thinking', 'working', 'completed'];
                    const randomStatus = statuses[Math.floor(Math.random() * statuses.length)];
                    
                    randomNode.status = randomStatus;
                    
                    // 노드 상태 색상 업데이트
                    nodes.select("circle:nth-child(1)")
                        .attr("stroke", d => config.status_colors[d.status]);
                    
                    nodes.select("circle:nth-child(3)")
                        .attr("fill", d => config.status_colors[d.status]);
                    
                    // 상태 업데이트
                    updateStatus();
                }}, 3000);
            </script>
        </body>
        </html>
        """
        
        return html_template
    
    def simulate_message_flow(self):
        """메시지 흐름 시뮬레이션"""
        # 랜덤 노드 선택
        node_ids = list(self.nodes.keys())
        if len(node_ids) < 2:
            return
        
        source_id = random.choice(node_ids)
        target_id = random.choice([n for n in node_ids if n != source_id])
        
        # 메시지 전송
        message_id = self.send_message(
            source_id,
            target_id,
            "analysis_request",
            {
                "task": "data_analysis",
                "priority": "high",
                "timestamp": time.time()
            }
        )
        
        return message_id
    
    def get_network_stats(self) -> Dict[str, Any]:
        """네트워크 통계 반환"""
        active_nodes = len([n for n in self.nodes.values() if n.status != NodeStatus.IDLE])
        active_connections = len([c for c in self.connections.values() if c.active])
        total_messages = sum(n.message_count for n in self.nodes.values())
        
        return {
            "total_nodes": len(self.nodes),
            "active_nodes": active_nodes,
            "total_connections": len(self.connections),
            "active_connections": active_connections,
            "total_messages": total_messages,
            "message_flows": len(self.message_flows),
            "node_types": {node_type.value: len([n for n in self.nodes.values() if n.type == node_type]) 
                          for node_type in NodeType},
            "connection_types": {conn_type.value: len([c for c in self.connections.values() if c.type == conn_type]) 
                               for conn_type in ConnectionType}
        }

# 싱글톤 인스턴스
_cursor_collaboration_network = None

def get_cursor_collaboration_network() -> CursorCollaborationNetwork:
    """Cursor 협업 네트워크 싱글톤 인스턴스 반환"""
    global _cursor_collaboration_network
    if _cursor_collaboration_network is None:
        _cursor_collaboration_network = CursorCollaborationNetwork()
    return _cursor_collaboration_network

def render_collaboration_network():
    """협업 네트워크 렌더링"""
    network = get_cursor_collaboration_network()
    
    # D3.js 시각화 HTML 생성
    html_content = network.get_d3_visualization()
    
    # Streamlit 컴포넌트로 렌더링
    components.html(html_content, height=700)
    
    return network 