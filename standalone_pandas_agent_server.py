#!/usr/bin/env python3
"""
🐼 Standalone Pandas Agent HTTP Server

A2A 없이 독립적으로 작동하는 자연어 데이터 분석 서버
CherryAI v10.0 Ultimate Integration - 핵심 기능 데모

Author: CherryAI Team
License: MIT License
"""

import asyncio
import json
import logging
import os
import sys
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 핵심 컴포넌트 임포트
try:
    from a2a_ds_servers.pandas_agent.pandas_agent_server import PandasAgentCore
    from a2a_ds_servers.pandas_agent.multi_dataframe_handler import MultiDataFrameHandler
    from a2a_ds_servers.pandas_agent.natural_language_processor import NaturalLanguageProcessor
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 핵심 컴포넌트 임포트 실패: {e}")
    COMPONENTS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # CORS 활성화

# 전역 변수
pandas_agent = None
data_handler = None
nlp_processor = None

def initialize_components():
    """핵심 컴포넌트 초기화"""
    global pandas_agent, data_handler, nlp_processor
    
    if not COMPONENTS_AVAILABLE:
        logger.error("❌ 핵심 컴포넌트를 사용할 수 없습니다")
        return False
    
    try:
        pandas_agent = PandasAgentCore()
        data_handler = MultiDataFrameHandler()
        nlp_processor = NaturalLanguageProcessor()
        
        logger.info("✅ Pandas Agent 핵심 컴포넌트 초기화 완료")
        return True
    except Exception as e:
        logger.error(f"❌ 컴포넌트 초기화 실패: {e}")
        return False

@app.route('/')
def home():
    """홈 페이지"""
    return jsonify({
        "name": "CherryAI Pandas Agent",
        "version": "10.0",
        "description": "자연어 기반 데이터 분석 서버",
        "status": "running",
        "endpoints": {
            "upload": "/api/upload",
            "query": "/api/query", 
            "dataframes": "/api/dataframes",
            "health": "/api/health"
        },
        "features": [
            "자연어 데이터 분석",
            "멀티 데이터프레임 처리",
            "관계 발견 및 병합",
            "한국어 지원"
        ]
    })

@app.route('/api/health')
def health_check():
    """서버 상태 확인"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "pandas_agent": pandas_agent is not None,
            "data_handler": data_handler is not None,
            "nlp_processor": nlp_processor is not None
        }
    })

@app.route('/api/upload', methods=['POST'])
def upload_data():
    """데이터 업로드"""
    try:
        if not pandas_agent:
            return jsonify({"error": "Pandas Agent가 초기화되지 않았습니다"}), 500
        
        # JSON 데이터로 DataFrame 생성
        if request.is_json:
            data = request.get_json()
            
            if 'data' in data:
                # 직접 데이터 전달
                df = pd.DataFrame(data['data'])
                name = data.get('name', f'dataset_{len(pandas_agent.dataframes)}')
                description = data.get('description', f'업로드된 데이터셋 {name}')
            else:
                # 전체 JSON을 DataFrame으로 변환
                df = pd.DataFrame(data)
                name = f'dataset_{len(pandas_agent.dataframes)}'
                description = '업로드된 JSON 데이터'
        else:
            return jsonify({"error": "JSON 데이터가 필요합니다"}), 400
        
        # DataFrame 추가
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        df_id = loop.run_until_complete(pandas_agent.add_dataframe(df, name, description))
        loop.close()
        
        return jsonify({
            "success": True,
            "dataframe_id": df_id,
            "name": name,
            "shape": df.shape,
            "columns": list(df.columns),
            "preview": df.head().to_dict('records')
        })
        
    except Exception as e:
        logger.error(f"❌ 데이터 업로드 실패: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/query', methods=['POST'])
def process_query():
    """자연어 쿼리 처리"""
    try:
        if not pandas_agent:
            return jsonify({"error": "Pandas Agent가 초기화되지 않았습니다"}), 500
        
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "쿼리가 필요합니다"}), 400
        
        query = data['query']
        
        # 자연어 쿼리 처리
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(pandas_agent.process_natural_language_query(query))
        loop.close()
        
        return jsonify({
            "success": True,
            "query": query,
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "dataframes_count": len(pandas_agent.dataframes)
        })
        
    except Exception as e:
        logger.error(f"❌ 쿼리 처리 실패: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/dataframes')
def list_dataframes():
    """등록된 데이터프레임 목록"""
    try:
        if not pandas_agent:
            return jsonify({"error": "Pandas Agent가 초기화되지 않았습니다"}), 500
        
        dataframes_info = []
        for i, metadata in enumerate(pandas_agent.dataframe_metadata):
            df = pandas_agent.dataframes[i]
            info = {
                "id": metadata['id'],
                "name": metadata['name'],
                "description": metadata['description'],
                "shape": metadata['shape'],
                "columns": metadata['columns'],
                "dtypes": metadata['dtypes'],
                "memory_usage": metadata['memory_usage'],
                "null_counts": metadata['null_counts'],
                "created_at": metadata['created_at'],
                "preview": df.head(3).to_dict('records') if not df.empty else []
            }
            dataframes_info.append(info)
        
        return jsonify({
            "success": True,
            "count": len(dataframes_info),
            "dataframes": dataframes_info
        })
        
    except Exception as e:
        logger.error(f"❌ 데이터프레임 목록 조회 실패: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/sample-data')
def create_sample_data():
    """샘플 데이터 생성 및 추가"""
    try:
        if not pandas_agent:
            return jsonify({"error": "Pandas Agent가 초기화되지 않았습니다"}), 500
        
        # 샘플 데이터 생성
        np.random.seed(42)
        sample_data = {
            'product': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B'],
            'sales': np.random.randint(100, 1000, 8),
            'month': np.random.choice(['Jan', 'Feb', 'Mar'], 8),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 8),
            'profit': np.random.randint(10, 100, 8)
        }
        
        df = pd.DataFrame(sample_data)
        
        # DataFrame 추가
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        df_id = loop.run_until_complete(pandas_agent.add_dataframe(
            df, 
            "sample_sales_data", 
            "판매 데이터 샘플"
        ))
        loop.close()
        
        return jsonify({
            "success": True,
            "message": "샘플 데이터가 생성되었습니다",
            "dataframe_id": df_id,
            "shape": df.shape,
            "preview": df.to_dict('records')
        })
        
    except Exception as e:
        logger.error(f"❌ 샘플 데이터 생성 실패: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/conversation')
def get_conversation():
    """대화 기록 조회"""
    try:
        if not pandas_agent:
            return jsonify({"error": "Pandas Agent가 초기화되지 않았습니다"}), 500
        
        history = pandas_agent.get_conversation_history()
        
        return jsonify({
            "success": True,
            "conversation_count": len(history),
            "history": history
        })
        
    except Exception as e:
        logger.error(f"❌ 대화 기록 조회 실패: {e}")
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "엔드포인트를 찾을 수 없습니다"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "서버 내부 오류가 발생했습니다"}), 500

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CherryAI Pandas Agent 독립 서버")
    parser.add_argument("--host", default="0.0.0.0", help="서버 호스트")
    parser.add_argument("--port", type=int, default=8080, help="서버 포트")
    parser.add_argument("--debug", action="store_true", help="디버그 모드")
    
    args = parser.parse_args()
    
    # 컴포넌트 초기화
    if not initialize_components():
        print("❌ 서버 초기화 실패")
        sys.exit(1)
    
    # 서버 정보 출력
    print("🚀 CherryAI Pandas Agent 독립 서버 시작")
    print(f"📍 주소: http://{args.host}:{args.port}")
    print(f"🔧 API 문서: http://{args.host}:{args.port}/")
    print(f"🔍 상태 확인: http://{args.host}:{args.port}/api/health")
    print()
    print("📋 사용 가능한 API:")
    print("  • POST /api/upload        - 데이터 업로드")
    print("  • POST /api/query         - 자연어 쿼리")
    print("  • GET  /api/dataframes    - 데이터프레임 목록")
    print("  • GET  /api/sample-data   - 샘플 데이터 생성")
    print("  • GET  /api/conversation  - 대화 기록")
    print()
    
    try:
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n🛑 서버가 종료되었습니다.")
    except Exception as e:
        print(f"❌ 서버 실행 오류: {e}")
        sys.exit(1) 