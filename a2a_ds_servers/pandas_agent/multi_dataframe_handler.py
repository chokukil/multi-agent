"""
🔄 Multi-DataFrame Handler

멀티 데이터프레임 처리 및 관리 시스템
A2A SDK를 통한 에이전트간 데이터 교환 지원

Author: CherryAI Team  
License: MIT License
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
import json
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFrameRegistry:
    """데이터프레임 중앙 레지스트리"""
    
    def __init__(self):
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.metadata: Dict[str, Dict] = {}
        self.relationships: Dict[str, List[str]] = {}
        self.created_at = datetime.now()
    
    def register_dataframe(self, df: pd.DataFrame, name: str = None, 
                          description: str = None, source: str = None) -> str:
        """데이터프레임 등록"""
        df_id = name or f"df_{uuid.uuid4().hex[:8]}"
        
        # 메타데이터 생성
        metadata = {
            "id": df_id,
            "name": name or df_id,
            "description": description or f"데이터프레임 {df_id}",
            "source": source or "unknown",
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "memory_usage": df.memory_usage(deep=True).sum(),
            "null_counts": df.isnull().sum().to_dict(),
            "created_at": datetime.now().isoformat(),
            "schema_hash": self._compute_schema_hash(df)
        }
        
        self.dataframes[df_id] = df
        self.metadata[df_id] = metadata
        self.relationships[df_id] = []
        
        logger.info(f"✅ 데이터프레임 등록: {df_id} (shape: {df.shape})")
        return df_id
    
    def _compute_schema_hash(self, df: pd.DataFrame) -> str:
        """스키마 해시 계산"""
        schema_str = f"{list(df.columns)}_{list(df.dtypes)}"
        return str(hash(schema_str))
    
    def get_dataframe(self, df_id: str) -> Optional[pd.DataFrame]:
        """데이터프레임 조회"""
        return self.dataframes.get(df_id)
    
    def get_metadata(self, df_id: str) -> Optional[Dict]:
        """메타데이터 조회"""
        return self.metadata.get(df_id)
    
    def list_dataframes(self) -> List[str]:
        """등록된 데이터프레임 목록"""
        return list(self.dataframes.keys())
    
    def find_similar_schemas(self, df_id: str) -> List[str]:
        """유사한 스키마의 데이터프레임 찾기"""
        if df_id not in self.metadata:
            return []
        
        target_hash = self.metadata[df_id]["schema_hash"]
        similar = []
        
        for other_id, meta in self.metadata.items():
            if other_id != df_id and meta["schema_hash"] == target_hash:
                similar.append(other_id)
        
        return similar
    
    def add_relationship(self, df_id1: str, df_id2: str, relationship_type: str = "related"):
        """데이터프레임 간 관계 추가"""
        if df_id1 in self.relationships:
            self.relationships[df_id1].append({"target": df_id2, "type": relationship_type})
        if df_id2 in self.relationships:
            self.relationships[df_id2].append({"target": df_id1, "type": relationship_type})
        
        logger.info(f"🔗 관계 추가: {df_id1} ↔ {df_id2} ({relationship_type})")


class MultiDataFrameHandler:
    """멀티 데이터프레임 핸들러"""
    
    def __init__(self):
        self.registry = DataFrameRegistry()
        self.current_context: List[str] = []  # 현재 작업 컨텍스트의 데이터프레임들
        
    async def add_dataframe(self, df: pd.DataFrame, **kwargs) -> str:
        """데이터프레임 추가"""
        df_id = self.registry.register_dataframe(df, **kwargs)
        
        # 자동 관계 발견
        await self._discover_relationships(df_id)
        
        return df_id
    
    async def _discover_relationships(self, df_id: str):
        """데이터프레임 간 관계 자동 발견"""
        try:
            df = self.registry.get_dataframe(df_id)
            if df is None:
                return
            
            # 1. 스키마 유사성 기반 관계
            similar_schemas = self.registry.find_similar_schemas(df_id)
            for similar_id in similar_schemas:
                self.registry.add_relationship(df_id, similar_id, "similar_schema")
            
            # 2. 컬럼명 기반 관계 발견
            await self._discover_column_relationships(df_id, df)
            
        except Exception as e:
            logger.warning(f"⚠️ 관계 발견 실패: {e}")
    
    async def _discover_column_relationships(self, df_id: str, df: pd.DataFrame):
        """컬럼명 기반 관계 발견"""
        df_columns = set(df.columns)
        
        for other_id in self.registry.list_dataframes():
            if other_id == df_id:
                continue
                
            other_df = self.registry.get_dataframe(other_id)
            if other_df is None:
                continue
            
            other_columns = set(other_df.columns)
            
            # 공통 컬럼 비율 계산
            common_columns = df_columns.intersection(other_columns)
            if common_columns:
                similarity_ratio = len(common_columns) / len(df_columns.union(other_columns))
                
                if similarity_ratio > 0.3:  # 30% 이상 공통 컬럼
                    relationship_type = "high_column_overlap" if similarity_ratio > 0.7 else "medium_column_overlap"
                    self.registry.add_relationship(df_id, other_id, relationship_type)
    
    def get_context_dataframes(self) -> List[pd.DataFrame]:
        """현재 컨텍스트의 데이터프레임들 반환"""
        return [self.registry.get_dataframe(df_id) for df_id in self.current_context 
                if self.registry.get_dataframe(df_id) is not None]
    
    def set_context(self, df_ids: List[str]):
        """작업 컨텍스트 설정"""
        # 유효한 데이터프레임 ID만 필터링
        valid_ids = [df_id for df_id in df_ids if df_id in self.registry.dataframes]
        self.current_context = valid_ids
        logger.info(f"📋 컨텍스트 설정: {len(valid_ids)}개 데이터프레임")
    
    def add_to_context(self, df_id: str):
        """컨텍스트에 데이터프레임 추가"""
        if df_id in self.registry.dataframes and df_id not in self.current_context:
            self.current_context.append(df_id)
            logger.info(f"➕ 컨텍스트에 추가: {df_id}")
    
    def remove_from_context(self, df_id: str):
        """컨텍스트에서 데이터프레임 제거"""
        if df_id in self.current_context:
            self.current_context.remove(df_id)
            logger.info(f"➖ 컨텍스트에서 제거: {df_id}")
    
    async def merge_dataframes(self, df_ids: List[str], how: str = 'inner', 
                             on: Optional[Union[str, List[str]]] = None) -> str:
        """데이터프레임 병합"""
        try:
            if len(df_ids) < 2:
                raise ValueError("최소 2개의 데이터프레임이 필요합니다.")
            
            dataframes = []
            for df_id in df_ids:
                df = self.registry.get_dataframe(df_id)
                if df is None:
                    raise ValueError(f"데이터프레임을 찾을 수 없습니다: {df_id}")
                dataframes.append(df)
            
            # 순차적 병합
            result_df = dataframes[0]
            merge_info = [self.registry.get_metadata(df_ids[0])['name']]
            
            for i, df in enumerate(dataframes[1:], 1):
                if on is None:
                    # 공통 컬럼 자동 찾기
                    common_cols = list(set(result_df.columns).intersection(set(df.columns)))
                    if not common_cols:
                        raise ValueError(f"병합할 공통 컬럼이 없습니다: {df_ids[0]} ↔ {df_ids[i]}")
                    merge_on = common_cols
                else:
                    merge_on = on
                
                result_df = pd.merge(result_df, df, on=merge_on, how=how)
                merge_info.append(self.registry.get_metadata(df_ids[i])['name'])
            
            # 병합 결과 등록
            merged_id = await self.add_dataframe(
                result_df,
                name=f"merged_{'_'.join(merge_info[:3])}",  # 이름 길이 제한
                description=f"병합된 데이터프레임: {' + '.join(merge_info)}",
                source="merge_operation"
            )
            
            logger.info(f"✅ 데이터프레임 병합 완료: {merged_id} (shape: {result_df.shape})")
            return merged_id
            
        except Exception as e:
            logger.error(f"❌ 데이터프레임 병합 실패: {e}")
            raise
    
    async def concat_dataframes(self, df_ids: List[str], axis: int = 0, 
                              ignore_index: bool = True) -> str:
        """데이터프레임 연결"""
        try:
            dataframes = []
            concat_info = []
            
            for df_id in df_ids:
                df = self.registry.get_dataframe(df_id)
                if df is None:
                    raise ValueError(f"데이터프레임을 찾을 수 없습니다: {df_id}")
                dataframes.append(df)
                concat_info.append(self.registry.get_metadata(df_id)['name'])
            
            # 연결 실행
            result_df = pd.concat(dataframes, axis=axis, ignore_index=ignore_index)
            
            # 연결 결과 등록
            concat_id = await self.add_dataframe(
                result_df,
                name=f"concat_{'_'.join(concat_info[:3])}",
                description=f"연결된 데이터프레임: {' + '.join(concat_info)} (axis={axis})",
                source="concat_operation"
            )
            
            logger.info(f"✅ 데이터프레임 연결 완료: {concat_id} (shape: {result_df.shape})")
            return concat_id
            
        except Exception as e:
            logger.error(f"❌ 데이터프레임 연결 실패: {e}")
            raise
    
    def get_summary_report(self) -> str:
        """멀티 데이터프레임 요약 보고서"""
        total_dfs = len(self.registry.dataframes)
        if total_dfs == 0:
            return "📊 등록된 데이터프레임이 없습니다."
        
        # 통계 계산
        total_rows = sum(meta['shape'][0] for meta in self.registry.metadata.values())
        total_columns = sum(meta['shape'][1] for meta in self.registry.metadata.values())
        total_memory = sum(meta['memory_usage'] for meta in self.registry.metadata.values())
        
        # 관계 통계
        total_relationships = sum(len(rels) for rels in self.registry.relationships.values()) // 2
        
        report = f"""# 📊 **멀티 데이터프레임 요약 보고서**

## 🔢 **전체 통계**
- **총 데이터프레임**: {total_dfs}개
- **총 행 수**: {total_rows:,}행
- **총 컬럼 수**: {total_columns}개
- **총 메모리 사용량**: {total_memory / 1024**2:.1f} MB
- **데이터프레임 간 관계**: {total_relationships}개

## 📋 **등록된 데이터프레임**
"""
        
        for df_id, metadata in self.registry.metadata.items():
            in_context = "🟢" if df_id in self.current_context else "⚪"
            report += f"""
### {in_context} **{metadata['name']}** (`{df_id}`)
- **크기**: {metadata['shape'][0]:,}행 × {metadata['shape'][1]}열
- **메모리**: {metadata['memory_usage'] / 1024**2:.1f} MB
- **결측치**: {sum(metadata['null_counts'].values())}개
- **생성일**: {metadata['created_at'][:19]}
"""
        
        if self.current_context:
            report += f"\n## 🎯 **현재 작업 컨텍스트**\n"
            report += f"활성 데이터프레임: {len(self.current_context)}개\n"
            for df_id in self.current_context:
                meta = self.registry.get_metadata(df_id)
                if meta:
                    report += f"- **{meta['name']}** ({meta['shape'][0]:,}행)\n"
        
        return report
    
    def export_metadata(self) -> Dict[str, Any]:
        """메타데이터 내보내기"""
        return {
            "registry_metadata": self.registry.metadata,
            "relationships": self.registry.relationships,
            "current_context": self.current_context,
            "created_at": self.registry.created_at.isoformat(),
            "total_dataframes": len(self.registry.dataframes)
        }
    
    async def import_metadata(self, metadata: Dict[str, Any]):
        """메타데이터 가져오기 (데이터프레임 제외)"""
        try:
            self.registry.metadata.update(metadata.get("registry_metadata", {}))
            self.registry.relationships.update(metadata.get("relationships", {}))
            self.current_context = metadata.get("current_context", [])
            
            logger.info(f"✅ 메타데이터 가져오기 완료: {len(metadata.get('registry_metadata', {}))}개")
            
        except Exception as e:
            logger.error(f"❌ 메타데이터 가져오기 실패: {e}")
            raise 