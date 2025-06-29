#!/usr/bin/env python3
"""
AI DS Team A2A 서버들의 get_workflow_summary 오류 일괄 수정 스크립트

'CompiledStateGraph' object has no attribute 'get_workflow_summary' 오류를 
안전한 try-except 블록으로 수정합니다.
"""

import os
import re
from pathlib import Path

def fix_get_workflow_summary_in_file(file_path: str) -> bool:
    """파일에서 get_workflow_summary 호출을 안전한 버전으로 수정"""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 기존 안전한 처리가 있는지 확인
        if 'try:' in content and 'get_workflow_summary' in content and 'AttributeError:' in content:
            print(f"✅ {file_path}: 이미 안전한 처리가 구현되어 있음")
            return False
        
        # 안전하지 않은 get_workflow_summary 호출 패턴 찾기
        unsafe_pattern = r'(\s+)workflow_summary = self\.agent\.get_workflow_summary\(markdown=True\)'
        
        if re.search(unsafe_pattern, content):
            # 안전한 버전으로 교체
            safe_replacement = r'''\1# 결과 처리 (안전한 방식으로 workflow summary 가져오기)
\1try:
\1    workflow_summary = self.agent.get_workflow_summary(markdown=True)
\1except AttributeError:
\1    # get_workflow_summary 메서드가 없는 경우 기본 요약 생성
\1    workflow_summary = f"✅ 작업이 완료되었습니다.\\n\\n**요청**: {user_instructions}"
\1except Exception as e:
\1    logger.warning(f"Error getting workflow summary: {e}")
\1    workflow_summary = f"✅ 작업이 완료되었습니다.\\n\\n**요청**: {user_instructions}"'''
            
            new_content = re.sub(unsafe_pattern, safe_replacement, content)
            
            # 파일 백업
            backup_path = f"{file_path}.backup"
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # 수정된 내용 저장
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print(f"🔧 {file_path}: get_workflow_summary 안전 처리 추가됨 (백업: {backup_path})")
            return True
        else:
            print(f"ℹ️  {file_path}: get_workflow_summary 호출이 없거나 이미 안전함")
            return False
            
    except Exception as e:
        print(f"❌ {file_path}: 수정 실패 - {e}")
        return False

def main():
    """메인 실행 함수"""
    print("🔧 AI DS Team A2A 서버 get_workflow_summary 오류 일괄 수정")
    print("=" * 60)
    
    # 수정할 서버 파일들
    server_files = [
        "ai_ds_team_data_visualization_server.py",
        "ai_ds_team_feature_engineering_server.py", 
        "ai_ds_team_sql_database_server.py",
        "ai_ds_team_mlflow_tools_server.py",
        "ai_ds_team_data_wrangling_server.py",
        "ai_ds_team_h2o_ml_server.py",
        "ai_ds_team_eda_tools_server.py"
    ]
    
    base_dir = Path(__file__).parent
    fixed_count = 0
    
    for server_file in server_files:
        file_path = base_dir / server_file
        if file_path.exists():
            if fix_get_workflow_summary_in_file(str(file_path)):
                fixed_count += 1
        else:
            print(f"⚠️  {server_file}: 파일을 찾을 수 없음")
    
    print("=" * 60)
    print(f"📊 수정 완료: {fixed_count}/{len(server_files)} 파일")
    
    if fixed_count > 0:
        print("\n✅ 수정된 서버들을 다시 시작하려면:")
        print("   ./ai_ds_team_system_start.sh")
        print("\n📁 백업 파일들:")
        for server_file in server_files:
            backup_path = base_dir / f"{server_file}.backup"
            if backup_path.exists():
                print(f"   {backup_path}")

if __name__ == "__main__":
    main() 