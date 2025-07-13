#!/usr/bin/env python3
"""
CherryAI 캐시 정리 스크립트 (Python 버전)
하위폴더까지 재귀적으로 Python 캐시 파일들을 제거합니다.
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import List, Set


def get_cache_patterns() -> List[str]:
    """제거할 캐시 패턴들을 반환합니다."""
    return [
        "__pycache__",
        "*.pyc",
        "*.pyo",
        ".pytest_cache",
        ".coverage",
        "htmlcov",
        ".mypy_cache",
        "build",
        "dist",
        "*.egg-info",
        ".tox",
        ".ipynb_checkpoints",
        "*.log",
        ".DS_Store",
    ]


def should_skip_directory(dir_path: Path) -> bool:
    """특정 디렉토리를 건너뛸지 결정합니다."""
    skip_dirs = {
        ".venv",
        "venv",
        ".env",
        "env",
        "node_modules",
        ".git",
        ".svn",
        ".hg"
    }
    
    # 절대 경로에서 디렉토리명 확인
    for part in dir_path.parts:
        if part in skip_dirs:
            return True
    
    return False


def find_cache_items(root_path: Path, dry_run: bool = False) -> Set[Path]:
    """캐시 파일과 폴더들을 찾습니다."""
    cache_items = set()
    
    for root, dirs, files in os.walk(root_path):
        root_path_obj = Path(root)
        
        # 건너뛸 디렉토리 확인
        if should_skip_directory(root_path_obj):
            dirs.clear()  # 하위 디렉토리 탐색 중지
            continue
        
        # 폴더 검사
        dirs_to_remove = []
        for dir_name in dirs:
            dir_path = root_path_obj / dir_name
            
            # 가상환경 디렉토리 건너뛰기
            if dir_name in {".venv", "venv", ".env", "env", "node_modules"}:
                dirs_to_remove.append(dir_name)
                continue
            
            # 캐시 폴더 패턴 검사
            if (dir_name == "__pycache__" or 
                dir_name == ".pytest_cache" or 
                dir_name == ".mypy_cache" or
                dir_name == "build" or
                dir_name == "dist" or
                dir_name == ".tox" or
                dir_name == ".ipynb_checkpoints" or
                dir_name == "htmlcov" or
                dir_name.endswith(".egg-info")):
                
                cache_items.add(dir_path)
                dirs_to_remove.append(dir_name)
        
        # 하위 폴더 탐색에서 제거된 폴더 제외
        for dir_name in dirs_to_remove:
            if dir_name in dirs:
                dirs.remove(dir_name)
        
        # 파일 검사
        for file_name in files:
            file_path = root_path_obj / file_name
            
            # 캐시 파일 패턴 검사
            if (file_name.endswith(".pyc") or 
                file_name.endswith(".pyo") or
                file_name == ".coverage" or
                file_name.endswith(".log") or
                file_name == ".DS_Store"):
                
                cache_items.add(file_path)
    
    return cache_items


def remove_cache_items(cache_items: Set[Path], dry_run: bool = False) -> tuple[int, int]:
    """캐시 아이템들을 제거합니다."""
    removed_files = 0
    removed_dirs = 0
    
    for item in sorted(cache_items):
        try:
            if not item.exists():
                continue
                
            if dry_run:
                print(f"[DRY RUN] 제거 예정: {item}")
                if item.is_dir():
                    removed_dirs += 1
                else:
                    removed_files += 1
                continue
            
            if item.is_dir():
                print(f"🗑️  폴더 제거: {item}")
                shutil.rmtree(item)
                removed_dirs += 1
            else:
                print(f"🗑️  파일 제거: {item}")
                item.unlink()
                removed_files += 1
                
        except Exception as e:
            print(f"❌ 제거 실패: {item} - {e}")
    
    return removed_files, removed_dirs


def main():
    parser = argparse.ArgumentParser(description="CherryAI 캐시 정리 스크립트")
    parser.add_argument("--path", "-p", type=str, default=".", 
                       help="정리할 경로 (기본값: 현재 디렉토리)")
    parser.add_argument("--dry-run", "-d", action="store_true",
                       help="실제 삭제 없이 삭제 예정 파일만 표시")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="상세한 출력")
    parser.add_argument("--include-venv", action="store_true",
                       help="가상환경 디렉토리(.venv, venv 등)도 포함하여 정리")
    
    args = parser.parse_args()
    
    root_path = Path(args.path).resolve()
    
    print("🍒 CherryAI 캐시 정리 시작...")
    print(f"📂 대상 디렉토리: {root_path}")
    
    if args.dry_run:
        print("🔍 DRY RUN 모드: 실제 삭제는 수행하지 않습니다.")
    
    if not args.include_venv:
        print("🚫 가상환경 디렉토리(.venv, venv 등)는 제외됩니다.")
        print("   포함하려면 --include-venv 옵션을 사용하세요.")
    
    if not root_path.exists():
        print(f"❌ 경로가 존재하지 않습니다: {root_path}")
        return
    
    # 캐시 아이템 찾기
    print("🔍 캐시 파일과 폴더 검색 중...")
    cache_items = find_cache_items(root_path, args.dry_run)
    
    if not cache_items:
        print("✅ 제거할 캐시 파일이 없습니다.")
        return
    
    print(f"📊 총 {len(cache_items)}개의 캐시 아이템을 발견했습니다.")
    
    if args.verbose or args.dry_run:
        print("\n발견된 캐시 아이템들:")
        for item in sorted(cache_items):
            item_type = "폴더" if item.is_dir() else "파일"
            print(f"  - {item_type}: {item}")
    
    # 캐시 아이템 제거
    print("\n🗑️  캐시 제거 중...")
    removed_files, removed_dirs = remove_cache_items(cache_items, args.dry_run)
    
    print(f"\n✅ 캐시 정리 완료!")
    print(f"📊 제거된 파일: {removed_files}개")
    print(f"📊 제거된 폴더: {removed_dirs}개")
    
    if args.dry_run:
        print("\n💡 실제로 제거하려면 --dry-run 옵션 없이 다시 실행하세요.")
    else:
        print("🎉 모든 Python 캐시 파일과 폴더가 재귀적으로 제거되었습니다.")


if __name__ == "__main__":
    main() 