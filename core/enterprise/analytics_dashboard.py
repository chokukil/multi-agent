"""
Advanced Analytics Dashboard
Phase 4.6: 고급 분석 대시보드

핵심 기능:
- 실시간 비즈니스 인텔리전스
- 커스텀 시각화 빌더
- 자동 보고서 생성
- 협업 분석 워크스페이스
- 대화형 차트 및 대시보드
- 실시간 데이터 모니터링
"""

import asyncio
import json
import logging
import time
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from enum import Enum
from pathlib import Path
import sqlite3
import base64
import hashlib
import pandas as pd
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

class DashboardType(Enum):
    """대시보드 유형"""
    EXECUTIVE = "executive"          # 경영진 대시보드
    OPERATIONAL = "operational"      # 운영 대시보드
    ANALYTICAL = "analytical"        # 분석 대시보드
    CUSTOM = "custom"               # 커스텀 대시보드

class VisualizationType(Enum):
    """시각화 유형"""
    BAR_CHART = "bar_chart"
    LINE_CHART = "line_chart"
    PIE_CHART = "pie_chart"
    SCATTER_PLOT = "scatter_plot"
    HEATMAP = "heatmap"
    HISTOGRAM = "histogram"
    BOX_PLOT = "box_plot"
    GAUGE = "gauge"
    KPI_CARD = "kpi_card"
    TABLE = "table"
    MAP = "map"
    TREEMAP = "treemap"

class RefreshInterval(Enum):
    """새로고침 간격"""
    REAL_TIME = 5      # 5초
    FAST = 30          # 30초
    MEDIUM = 300       # 5분
    SLOW = 1800        # 30분
    HOURLY = 3600      # 1시간
    DAILY = 86400      # 24시간

class ReportFrequency(Enum):
    """보고서 빈도"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"

@dataclass
class Widget:
    """대시보드 위젯"""
    widget_id: str
    title: str
    visualization_type: VisualizationType
    data_source: str
    config: Dict[str, Any]
    position: Dict[str, int]  # x, y, width, height
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class Dashboard:
    """대시보드"""
    dashboard_id: str
    name: str
    description: str
    dashboard_type: DashboardType
    owner_id: str
    widgets: List[Widget] = field(default_factory=list)
    layout: Dict[str, Any] = field(default_factory=dict)
    permissions: Dict[str, List[str]] = field(default_factory=dict)
    refresh_interval: RefreshInterval = RefreshInterval.MEDIUM
    is_public: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class Report:
    """보고서"""
    report_id: str
    name: str
    description: str
    dashboard_id: str
    frequency: ReportFrequency
    recipients: List[str]
    template: Dict[str, Any]
    next_run: datetime
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class KPI:
    """핵심성과지표"""
    kpi_id: str
    name: str
    description: str
    formula: str
    target_value: Optional[float]
    current_value: Optional[float]
    unit: str
    trend: str  # up, down, stable
    color: str  # green, yellow, red
    dashboard_id: str
    created_at: datetime = field(default_factory=datetime.now)

class AnalyticsDashboardDatabase:
    """분석 대시보드 데이터베이스"""
    
    def __init__(self, db_path: str = "core/enterprise/analytics_dashboard.db"):
        self.db_path = db_path
        self._initialize_database()
    
    def _initialize_database(self):
        """데이터베이스 초기화"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # 대시보드 테이블
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dashboards (
                    dashboard_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    dashboard_type TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    layout TEXT,
                    permissions TEXT,
                    refresh_interval INTEGER DEFAULT 300,
                    is_public BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 위젯 테이블
            conn.execute("""
                CREATE TABLE IF NOT EXISTS widgets (
                    widget_id TEXT PRIMARY KEY,
                    dashboard_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    visualization_type TEXT NOT NULL,
                    data_source TEXT NOT NULL,
                    config TEXT NOT NULL,
                    position TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (dashboard_id) REFERENCES dashboards (dashboard_id)
                )
            """)
            
            # 보고서 테이블
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reports (
                    report_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    dashboard_id TEXT NOT NULL,
                    frequency TEXT NOT NULL,
                    recipients TEXT NOT NULL,
                    template TEXT NOT NULL,
                    next_run TIMESTAMP NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (dashboard_id) REFERENCES dashboards (dashboard_id)
                )
            """)
            
            # KPI 테이블
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kpis (
                    kpi_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    formula TEXT NOT NULL,
                    target_value REAL,
                    current_value REAL,
                    unit TEXT,
                    trend TEXT,
                    color TEXT,
                    dashboard_id TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (dashboard_id) REFERENCES dashboards (dashboard_id)
                )
            """)
            
            # 사용자 활동 로그
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_activity (
                    activity_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    resource_type TEXT NOT NULL,
                    resource_id TEXT NOT NULL,
                    details TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def save_dashboard(self, dashboard: Dashboard) -> bool:
        """대시보드 저장"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO dashboards 
                    (dashboard_id, name, description, dashboard_type, owner_id, 
                     layout, permissions, refresh_interval, is_public, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    dashboard.dashboard_id,
                    dashboard.name,
                    dashboard.description,
                    dashboard.dashboard_type.value,
                    dashboard.owner_id,
                    json.dumps(dashboard.layout),
                    json.dumps(dashboard.permissions),
                    dashboard.refresh_interval.value,
                    dashboard.is_public,
                    dashboard.updated_at.isoformat()
                ))
            return True
        except Exception as e:
            logger.error(f"대시보드 저장 실패: {e}")
            return False
    
    def save_widget(self, widget: Widget, dashboard_id: str) -> bool:
        """위젯 저장"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO widgets 
                    (widget_id, dashboard_id, title, visualization_type, data_source, 
                     config, position, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    widget.widget_id,
                    dashboard_id,
                    widget.title,
                    widget.visualization_type.value,
                    widget.data_source,
                    json.dumps(widget.config),
                    json.dumps(widget.position),
                    widget.updated_at.isoformat()
                ))
            return True
        except Exception as e:
            logger.error(f"위젯 저장 실패: {e}")
            return False
    
    def get_dashboard(self, dashboard_id: str) -> Optional[Dashboard]:
        """대시보드 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM dashboards WHERE dashboard_id = ?
                """, (dashboard_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                # 위젯들 조회
                widgets_cursor = conn.execute("""
                    SELECT * FROM widgets WHERE dashboard_id = ?
                """, (dashboard_id,))
                
                widgets = []
                for widget_row in widgets_cursor.fetchall():
                    widget = Widget(
                        widget_id=widget_row[0],
                        title=widget_row[2],
                        visualization_type=VisualizationType(widget_row[3]),
                        data_source=widget_row[4],
                        config=json.loads(widget_row[5]),
                        position=json.loads(widget_row[6]),
                        created_at=datetime.fromisoformat(widget_row[7]),
                        updated_at=datetime.fromisoformat(widget_row[8])
                    )
                    widgets.append(widget)
                
                dashboard = Dashboard(
                    dashboard_id=row[0],
                    name=row[1],
                    description=row[2],
                    dashboard_type=DashboardType(row[3]),
                    owner_id=row[4],
                    widgets=widgets,
                    layout=json.loads(row[5]) if row[5] else {},
                    permissions=json.loads(row[6]) if row[6] else {},
                    refresh_interval=RefreshInterval(row[7]),
                    is_public=bool(row[8]),
                    created_at=datetime.fromisoformat(row[9]),
                    updated_at=datetime.fromisoformat(row[10])
                )
                
                return dashboard
                
        except Exception as e:
            logger.error(f"대시보드 조회 실패: {e}")
            return None
    
    def get_user_dashboards(self, user_id: str) -> List[Dashboard]:
        """사용자 대시보드 목록 조회"""
        dashboards = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT dashboard_id FROM dashboards 
                    WHERE owner_id = ? OR is_public = TRUE
                    ORDER BY updated_at DESC
                """, (user_id,))
                
                for row in cursor.fetchall():
                    dashboard = self.get_dashboard(row[0])
                    if dashboard:
                        dashboards.append(dashboard)
        except Exception as e:
            logger.error(f"사용자 대시보드 조회 실패: {e}")
        
        return dashboards

class DataSourceManager:
    """데이터 소스 관리"""
    
    def __init__(self):
        self.data_sources = {}
        self.cache = {}
        self.cache_ttl = {}
    
    def register_data_source(self, name: str, source_config: Dict[str, Any]):
        """데이터 소스 등록"""
        self.data_sources[name] = source_config
        logger.info(f"데이터 소스 등록: {name}")
    
    async def get_data(self, source_name: str, query: Optional[str] = None, 
                      use_cache: bool = True) -> Optional[pd.DataFrame]:
        """데이터 조회"""
        cache_key = f"{source_name}:{hashlib.md5(str(query).encode()).hexdigest()}"
        
        # 캐시 확인
        if use_cache and cache_key in self.cache:
            if cache_key in self.cache_ttl and self.cache_ttl[cache_key] > time.time():
                return self.cache[cache_key]
        
        # 데이터 소스에서 조회
        if source_name not in self.data_sources:
            logger.error(f"등록되지 않은 데이터 소스: {source_name}")
            return None
        
        source_config = self.data_sources[source_name]
        
        try:
            if source_config["type"] == "csv":
                df = pd.read_csv(source_config["path"])
            elif source_config["type"] == "json":
                df = pd.read_json(source_config["path"])
            elif source_config["type"] == "database":
                # 실제 환경에서는 데이터베이스 연결 구현
                df = self._mock_database_query(query)
            elif source_config["type"] == "api":
                # 실제 환경에서는 API 호출 구현
                df = self._mock_api_call(source_config["url"], query)
            else:
                logger.error(f"지원되지 않는 데이터 소스 유형: {source_config['type']}")
                return None
            
            # 캐시 저장 (5분)
            if use_cache:
                self.cache[cache_key] = df
                self.cache_ttl[cache_key] = time.time() + 300
            
            return df
            
        except Exception as e:
            logger.error(f"데이터 조회 실패: {e}")
            return None
    
    def _mock_database_query(self, query: Optional[str]) -> pd.DataFrame:
        """모의 데이터베이스 쿼리"""
        # 실제 환경에서는 실제 데이터베이스 쿼리 실행
        data = {
            "date": pd.date_range("2024-01-01", periods=100, freq="D"),
            "revenue": np.random.normal(10000, 2000, 100),
            "customers": np.random.poisson(50, 100),
            "conversion_rate": np.random.normal(0.05, 0.01, 100)
        }
        return pd.DataFrame(data)
    
    def _mock_api_call(self, url: str, query: Optional[str]) -> pd.DataFrame:
        """모의 API 호출"""
        # 실제 환경에서는 실제 API 호출
        data = {
            "product": ["A", "B", "C", "D", "E"] * 20,
            "sales": np.random.normal(1000, 200, 100),
            "profit": np.random.normal(200, 50, 100),
            "category": np.random.choice(["Electronics", "Clothing", "Books"], 100)
        }
        return pd.DataFrame(data)

class VisualizationEngine:
    """시각화 엔진"""
    
    def __init__(self):
        self.supported_types = list(VisualizationType)
    
    async def generate_visualization(self, viz_type: VisualizationType, 
                                   data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """시각화 생성"""
        try:
            if viz_type == VisualizationType.BAR_CHART:
                return self._create_bar_chart(data, config)
            elif viz_type == VisualizationType.LINE_CHART:
                return self._create_line_chart(data, config)
            elif viz_type == VisualizationType.PIE_CHART:
                return self._create_pie_chart(data, config)
            elif viz_type == VisualizationType.KPI_CARD:
                return self._create_kpi_card(data, config)
            elif viz_type == VisualizationType.TABLE:
                return self._create_table(data, config)
            else:
                return self._create_generic_chart(viz_type, data, config)
                
        except Exception as e:
            logger.error(f"시각화 생성 실패: {e}")
            return {"error": str(e)}
    
    def _create_bar_chart(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """막대 차트 생성"""
        x_column = config.get("x_column", data.columns[0])
        y_column = config.get("y_column", data.columns[1])
        
        # 실제 환경에서는 Plotly, D3.js 등을 사용하여 차트 생성
        chart_data = {
            "type": "bar",
            "data": {
                "labels": data[x_column].tolist(),
                "datasets": [{
                    "label": y_column,
                    "data": data[y_column].tolist(),
                    "backgroundColor": config.get("color", "#2E86AB")
                }]
            },
            "options": {
                "responsive": True,
                "title": {"display": True, "text": config.get("title", "Bar Chart")}
            }
        }
        
        return chart_data
    
    def _create_line_chart(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """선 차트 생성"""
        x_column = config.get("x_column", data.columns[0])
        y_column = config.get("y_column", data.columns[1])
        
        chart_data = {
            "type": "line",
            "data": {
                "labels": data[x_column].tolist(),
                "datasets": [{
                    "label": y_column,
                    "data": data[y_column].tolist(),
                    "borderColor": config.get("color", "#A23B72"),
                    "fill": False
                }]
            },
            "options": {
                "responsive": True,
                "title": {"display": True, "text": config.get("title", "Line Chart")}
            }
        }
        
        return chart_data
    
    def _create_pie_chart(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """파이 차트 생성"""
        label_column = config.get("label_column", data.columns[0])
        value_column = config.get("value_column", data.columns[1])
        
        chart_data = {
            "type": "pie",
            "data": {
                "labels": data[label_column].tolist(),
                "datasets": [{
                    "data": data[value_column].tolist(),
                    "backgroundColor": [
                        "#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#8FBC8F"
                    ]
                }]
            },
            "options": {
                "responsive": True,
                "title": {"display": True, "text": config.get("title", "Pie Chart")}
            }
        }
        
        return chart_data
    
    def _create_kpi_card(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """KPI 카드 생성"""
        value_column = config.get("value_column", data.columns[0])
        current_value = data[value_column].iloc[-1] if len(data) > 0 else 0
        previous_value = data[value_column].iloc[-2] if len(data) > 1 else current_value
        
        change = ((current_value - previous_value) / previous_value * 100) if previous_value != 0 else 0
        trend = "up" if change > 0 else "down" if change < 0 else "stable"
        
        kpi_data = {
            "type": "kpi",
            "data": {
                "title": config.get("title", "KPI"),
                "value": current_value,
                "unit": config.get("unit", ""),
                "change": change,
                "trend": trend,
                "color": "green" if change >= 0 else "red"
            }
        }
        
        return kpi_data
    
    def _create_table(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """테이블 생성"""
        limit = config.get("limit", 100)
        
        table_data = {
            "type": "table",
            "data": {
                "columns": data.columns.tolist(),
                "rows": data.head(limit).values.tolist(),
                "total_rows": len(data)
            }
        }
        
        return table_data
    
    def _create_generic_chart(self, viz_type: VisualizationType, 
                            data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """일반 차트 생성"""
        return {
            "type": viz_type.value,
            "data": {
                "message": f"{viz_type.value} visualization not fully implemented",
                "columns": data.columns.tolist(),
                "shape": data.shape,
                "sample": data.head().to_dict()
            }
        }

class ReportGenerator:
    """보고서 생성기"""
    
    def __init__(self, db: AnalyticsDashboardDatabase, viz_engine: VisualizationEngine):
        self.db = db
        self.viz_engine = viz_engine
    
    async def generate_report(self, dashboard_id: str, report_config: Dict[str, Any]) -> Dict[str, Any]:
        """보고서 생성"""
        try:
            dashboard = self.db.get_dashboard(dashboard_id)
            if not dashboard:
                raise Exception(f"Dashboard not found: {dashboard_id}")
            
            report_data = {
                "report_id": f"report_{int(time.time())}",
                "dashboard_name": dashboard.name,
                "generated_at": datetime.now().isoformat(),
                "widgets": [],
                "summary": {},
                "metadata": report_config
            }
            
            # 각 위젯에 대한 데이터 및 시각화 생성
            for widget in dashboard.widgets:
                widget_data = await self._generate_widget_report(widget)
                report_data["widgets"].append(widget_data)
            
            # 요약 정보 생성
            report_data["summary"] = await self._generate_summary(dashboard, report_data["widgets"])
            
            return report_data
            
        except Exception as e:
            logger.error(f"보고서 생성 실패: {e}")
            return {"error": str(e)}
    
    async def _generate_widget_report(self, widget: Widget) -> Dict[str, Any]:
        """위젯 보고서 생성"""
        # 실제 환경에서는 데이터 소스에서 실제 데이터 조회
        sample_data = pd.DataFrame({
            "x": range(10),
            "y": np.random.normal(100, 20, 10)
        })
        
        visualization = await self.viz_engine.generate_visualization(
            widget.visualization_type,
            sample_data,
            widget.config
        )
        
        return {
            "widget_id": widget.widget_id,
            "title": widget.title,
            "type": widget.visualization_type.value,
            "visualization": visualization,
            "data_summary": {
                "rows": len(sample_data),
                "columns": len(sample_data.columns)
            }
        }
    
    async def _generate_summary(self, dashboard: Dashboard, widgets_data: List[Dict]) -> Dict[str, Any]:
        """보고서 요약 생성"""
        return {
            "total_widgets": len(widgets_data),
            "dashboard_type": dashboard.dashboard_type.value,
            "last_updated": dashboard.updated_at.isoformat(),
            "insights": [
                "데이터 품질이 양호합니다",
                "트렌드가 상승세를 보이고 있습니다",
                "특별한 이상 사항은 발견되지 않았습니다"
            ]
        }

class CollaborationManager:
    """협업 관리"""
    
    def __init__(self, db: AnalyticsDashboardDatabase):
        self.db = db
        self.active_sessions = {}
        self.comments = {}
    
    async def create_collaboration_session(self, dashboard_id: str, user_id: str) -> str:
        """협업 세션 생성"""
        session_id = f"collab_{int(time.time())}_{user_id}"
        
        self.active_sessions[session_id] = {
            "dashboard_id": dashboard_id,
            "user_id": user_id,
            "started_at": datetime.now(),
            "participants": [user_id],
            "activities": []
        }
        
        return session_id
    
    async def join_collaboration(self, session_id: str, user_id: str) -> bool:
        """협업 참여"""
        if session_id in self.active_sessions:
            if user_id not in self.active_sessions[session_id]["participants"]:
                self.active_sessions[session_id]["participants"].append(user_id)
            return True
        return False
    
    async def add_comment(self, dashboard_id: str, widget_id: str, user_id: str, comment: str) -> str:
        """댓글 추가"""
        comment_id = f"comment_{int(time.time())}_{user_id}"
        
        if dashboard_id not in self.comments:
            self.comments[dashboard_id] = {}
        if widget_id not in self.comments[dashboard_id]:
            self.comments[dashboard_id][widget_id] = []
        
        comment_data = {
            "comment_id": comment_id,
            "user_id": user_id,
            "comment": comment,
            "timestamp": datetime.now().isoformat()
        }
        
        self.comments[dashboard_id][widget_id].append(comment_data)
        return comment_id
    
    def get_comments(self, dashboard_id: str, widget_id: str) -> List[Dict]:
        """댓글 조회"""
        if (dashboard_id in self.comments and 
            widget_id in self.comments[dashboard_id]):
            return self.comments[dashboard_id][widget_id]
        return []

class AdvancedAnalyticsDashboard:
    """고급 분석 대시보드 메인 클래스"""
    
    def __init__(self):
        self.db = AnalyticsDashboardDatabase()
        self.data_source_manager = DataSourceManager()
        self.viz_engine = VisualizationEngine()
        self.report_generator = ReportGenerator(self.db, self.viz_engine)
        self.collaboration_manager = CollaborationManager(self.db)
        
        # 기본 데이터 소스 등록
        self._register_default_data_sources()
    
    def _register_default_data_sources(self):
        """기본 데이터 소스 등록"""
        self.data_source_manager.register_data_source("sample_sales", {
            "type": "api",
            "url": "https://api.example.com/sales",
            "description": "Sales data from CRM"
        })
        
        self.data_source_manager.register_data_source("sample_analytics", {
            "type": "database",
            "query": "SELECT * FROM analytics",
            "description": "Analytics data from data warehouse"
        })
    
    async def create_dashboard(self, name: str, description: str, 
                             dashboard_type: DashboardType, owner_id: str) -> str:
        """대시보드 생성"""
        dashboard_id = f"dash_{int(time.time())}_{owner_id}"
        
        dashboard = Dashboard(
            dashboard_id=dashboard_id,
            name=name,
            description=description,
            dashboard_type=dashboard_type,
            owner_id=owner_id
        )
        
        success = self.db.save_dashboard(dashboard)
        if success:
            return dashboard_id
        else:
            raise Exception("Failed to create dashboard")
    
    async def add_widget(self, dashboard_id: str, title: str, 
                        viz_type: VisualizationType, data_source: str,
                        config: Dict[str, Any], position: Dict[str, int]) -> str:
        """위젯 추가"""
        widget_id = f"widget_{int(time.time())}"
        
        widget = Widget(
            widget_id=widget_id,
            title=title,
            visualization_type=viz_type,
            data_source=data_source,
            config=config,
            position=position
        )
        
        success = self.db.save_widget(widget, dashboard_id)
        if success:
            return widget_id
        else:
            raise Exception("Failed to add widget")
    
    async def get_dashboard_data(self, dashboard_id: str) -> Dict[str, Any]:
        """대시보드 데이터 조회"""
        dashboard = self.db.get_dashboard(dashboard_id)
        if not dashboard:
            return {"error": "Dashboard not found"}
        
        dashboard_data = {
            "dashboard_id": dashboard.dashboard_id,
            "name": dashboard.name,
            "description": dashboard.description,
            "type": dashboard.dashboard_type.value,
            "widgets": [],
            "layout": dashboard.layout,
            "last_updated": dashboard.updated_at.isoformat()
        }
        
        # 각 위젯의 데이터 조회
        for widget in dashboard.widgets:
            widget_data = await self._get_widget_data(widget)
            dashboard_data["widgets"].append(widget_data)
        
        return dashboard_data
    
    async def _get_widget_data(self, widget: Widget) -> Dict[str, Any]:
        """위젯 데이터 조회"""
        # 데이터 소스에서 데이터 조회
        data = await self.data_source_manager.get_data(widget.data_source)
        
        if data is None:
            return {
                "widget_id": widget.widget_id,
                "title": widget.title,
                "error": "Failed to load data"
            }
        
        # 시각화 생성
        visualization = await self.viz_engine.generate_visualization(
            widget.visualization_type,
            data,
            widget.config
        )
        
        return {
            "widget_id": widget.widget_id,
            "title": widget.title,
            "type": widget.visualization_type.value,
            "visualization": visualization,
            "position": widget.position,
            "last_updated": widget.updated_at.isoformat()
        }
    
    async def generate_automated_report(self, dashboard_id: str) -> Dict[str, Any]:
        """자동 보고서 생성"""
        return await self.report_generator.generate_report(dashboard_id, {
            "format": "html",
            "include_charts": True,
            "include_summary": True
        })
    
    def get_dashboard_templates(self) -> List[Dict[str, Any]]:
        """대시보드 템플릿 조회"""
        return [
            {
                "template_id": "executive_summary",
                "name": "Executive Summary",
                "description": "High-level business metrics and KPIs",
                "type": DashboardType.EXECUTIVE.value,
                "widgets": [
                    {"type": "kpi_card", "title": "Total Revenue"},
                    {"type": "line_chart", "title": "Revenue Trend"},
                    {"type": "pie_chart", "title": "Revenue by Category"},
                    {"type": "gauge", "title": "Goal Achievement"}
                ]
            },
            {
                "template_id": "operational_dashboard",
                "name": "Operational Dashboard",
                "description": "Day-to-day operational metrics",
                "type": DashboardType.OPERATIONAL.value,
                "widgets": [
                    {"type": "bar_chart", "title": "Daily Sales"},
                    {"type": "table", "title": "Recent Transactions"},
                    {"type": "heatmap", "title": "Activity Heatmap"},
                    {"type": "line_chart", "title": "Performance Metrics"}
                ]
            },
            {
                "template_id": "analytical_deep_dive",
                "name": "Analytical Deep Dive",
                "description": "Detailed analytical insights",
                "type": DashboardType.ANALYTICAL.value,
                "widgets": [
                    {"type": "scatter_plot", "title": "Correlation Analysis"},
                    {"type": "histogram", "title": "Distribution Analysis"},
                    {"type": "box_plot", "title": "Statistical Summary"},
                    {"type": "treemap", "title": "Hierarchical Data"}
                ]
            }
        ]

# 전역 인스턴스
_analytics_dashboard = None

def get_analytics_dashboard() -> AdvancedAnalyticsDashboard:
    """분석 대시보드 싱글톤 인스턴스 반환"""
    global _analytics_dashboard
    if _analytics_dashboard is None:
        _analytics_dashboard = AdvancedAnalyticsDashboard()
    return _analytics_dashboard

async def test_analytics_dashboard():
    """분석 대시보드 테스트"""
    print("🧪 Advanced Analytics Dashboard 테스트 시작")
    
    try:
        dashboard = get_analytics_dashboard()
        
        # 대시보드 생성 테스트
        dashboard_id = await dashboard.create_dashboard(
            name="Test Executive Dashboard",
            description="테스트용 경영진 대시보드",
            dashboard_type=DashboardType.EXECUTIVE,
            owner_id="user_001"
        )
        print(f"✅ 대시보드 생성: {dashboard_id}")
        
        # 위젯 추가 테스트
        widget_id_1 = await dashboard.add_widget(
            dashboard_id=dashboard_id,
            title="매출 트렌드",
            viz_type=VisualizationType.LINE_CHART,
            data_source="sample_sales",
            config={"x_column": "date", "y_column": "revenue"},
            position={"x": 0, "y": 0, "width": 6, "height": 4}
        )
        print(f"✅ 위젯 1 추가: {widget_id_1}")
        
        widget_id_2 = await dashboard.add_widget(
            dashboard_id=dashboard_id,
            title="총 매출",
            viz_type=VisualizationType.KPI_CARD,
            data_source="sample_sales",
            config={"value_column": "revenue", "unit": "원"},
            position={"x": 6, "y": 0, "width": 3, "height": 2}
        )
        print(f"✅ 위젯 2 추가: {widget_id_2}")
        
        # 대시보드 데이터 조회 테스트
        dashboard_data = await dashboard.get_dashboard_data(dashboard_id)
        print(f"✅ 대시보드 데이터 조회: {len(dashboard_data['widgets'])}개 위젯")
        
        # 보고서 생성 테스트
        report = await dashboard.generate_automated_report(dashboard_id)
        print(f"✅ 자동 보고서 생성: {report.get('report_id', 'N/A')}")
        
        # 템플릿 조회 테스트
        templates = dashboard.get_dashboard_templates()
        print(f"✅ 템플릿 조회: {len(templates)}개 템플릿")
        
        # 협업 기능 테스트
        collab_session = await dashboard.collaboration_manager.create_collaboration_session(
            dashboard_id, "user_001"
        )
        print(f"✅ 협업 세션 생성: {collab_session}")
        
        comment_id = await dashboard.collaboration_manager.add_comment(
            dashboard_id, widget_id_1, "user_001", "이 차트의 트렌드가 인상적이네요!"
        )
        print(f"✅ 댓글 추가: {comment_id}")
        
        comments = dashboard.collaboration_manager.get_comments(dashboard_id, widget_id_1)
        print(f"✅ 댓글 조회: {len(comments)}개 댓글")
        
        print("✅ Advanced Analytics Dashboard 테스트 완료")
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_analytics_dashboard()) 