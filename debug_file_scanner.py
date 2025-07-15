#!/usr/bin/env python3
"""
FileScanner 디버깅 스크립트

실제 파일 스캔 결과를 확인하여 왜 sample_sales_data.csv가 감지되지 않는지 분석
"""

import asyncio
import os
import sys
from pathlib import Path

# 현재 프로젝트 경로를 Python 경로에 추가
sys.path.append('.')
sys.path.append('./a2a_ds_servers')

from a2a_ds_servers.unified_data_system.utils.file_scanner import FileScanner

async def debug_file_scanner():
    """FileScanner 디버깅"""
    
    print("🔍 FileScanner 디버깅 시작...")
    print(f"📂 현재 작업 디렉토리: {os.getcwd()}")
    
    # FileScanner 초기화
    scanner = FileScanner()
    
    print(f"\n📋 FileScanner 기본 경로들:")
    for i, path in enumerate(scanner.base_paths, 1):
        abs_path = os.path.abspath(path)
        exists = os.path.exists(path)
        print(f"  {i}. {path}")
        print(f"     → 절대경로: {abs_path}")
        print(f"     → 존재여부: {'✅' if exists else '❌'}")
        
        if exists:
            try:
                files = list(Path(path).rglob('*.*'))
                print(f"     → 파일개수: {len(files)}개")
                for file in files[:3]:  # 첫 3개만 표시
                    print(f"       - {file}")
                if len(files) > 3:
                    print(f"       ... 및 {len(files)-3}개 더")
            except Exception as e:
                print(f"     → 스캔오류: {e}")
        print()
    
    # 실제 파일 스캔 실행
    print("🔄 실제 파일 스캔 실행...")
    found_files = await scanner.scan_data_files()
    
    print(f"\n📊 스캔 결과: {len(found_files)}개 파일 발견")
    for i, file_path in enumerate(found_files, 1):
        print(f"  {i}. {file_path}")
    
    # 특정 파일 검색
    print(f"\n🎯 'sample_sales_data.csv' 검색...")
    target_files = await scanner.find_files_by_pattern("sample_sales_data")
    
    print(f"📋 패턴 매칭 결과: {len(target_files)}개")
    for file_path in target_files:
        print(f"  ✅ {file_path}")
    
    # 수동으로 파일 존재 확인
    manual_paths = [
        "a2a_ds_servers/artifacts/data/sample_sales_data.csv",
        "./a2a_ds_servers/artifacts/data/sample_sales_data.csv",
        os.path.abspath("a2a_ds_servers/artifacts/data/sample_sales_data.csv")
    ]
    
    print(f"\n🔧 수동 파일 존재 확인:")
    for path in manual_paths:
        exists = os.path.exists(path)
        print(f"  {'✅' if exists else '❌'} {path}")

if __name__ == "__main__":
    asyncio.run(debug_file_scanner()) 