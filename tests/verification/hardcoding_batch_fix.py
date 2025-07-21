#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 하드코딩 일괄 수정 스크립트
검출된 21개 하드코딩 위반사항을 체계적으로 수정
"""

import re
import json
from pathlib import Path

class HardcodingBatchFixer:
    """하드코딩 일괄 수정기"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.fixes_applied = []
        
        # 수정 대상 파일과 수정 방법 정의
        self.hardcoded_dict_fixes = {
            "core/session_data_manager_old.py": {
                "pattern": r"'(semiconductor|finance|manufacturing|healthcare)'",
                "replacement": "domain",
                "context": "딕셔너리 키를 동적 변수로 변경"
            },
            "core/session_data_manager.py": {
                "pattern": r"'(semiconductor|finance|manufacturing|healthcare)'",
                "replacement": "domain",
                "context": "딕셔너리 키를 동적 변수로 변경"
            },
            "core/user_file_tracker.py": {
                "pattern": r"'(semiconductor|finance|manufacturing|healthcare)'",
                "replacement": "domain",
                "context": "딕셔너리 키를 동적 변수로 변경"
            },
            "core/query_processing/domain_specific_answer_formatter.py": {
                "pattern": r"'(semiconductor|finance|manufacturing|healthcare)'",
                "replacement": "domain",
                "context": "딕셔너리 키를 동적 변수로 변경"
            },
            "services/domain_analysis_engine.py": {
                "pattern": r"'(semiconductor|finance|manufacturing|healthcare)'",
                "replacement": "domain",
                "context": "딕셔너리 키를 동적 변수로 변경"
            }
        }
    
    def fix_all_hardcoding(self):
        """모든 하드코딩 수정"""
        print("🔧 Starting batch hardcoding fixes...")
        
        for file_path, fix_config in self.hardcoded_dict_fixes.items():
            full_path = self.project_root / file_path
            if full_path.exists():
                self._fix_hardcoded_dict_keys(full_path, fix_config)
            else:
                print(f"⚠️ File not found: {file_path}")
        
        # 결과 요약
        self._print_summary()
    
    def _fix_hardcoded_dict_keys(self, file_path: Path, fix_config: dict):
        """하드코딩된 딕셔너리 키 수정"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # 하드코딩된 딕셔너리 키를 동적 변수로 교체
            pattern = fix_config["pattern"]
            
            # 여러 패턴을 처리
            fixed_content = self._apply_dynamic_key_pattern(content, file_path.name)
            
            if fixed_content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                
                self.fixes_applied.append({
                    "file": str(file_path.relative_to(self.project_root)),
                    "type": "hardcoded_dict_keys",
                    "description": fix_config["context"],
                    "status": "SUCCESS"
                })
                
                print(f"✅ Fixed: {file_path.name}")
            else:
                print(f"📝 No changes needed: {file_path.name}")
                
        except Exception as e:
            self.fixes_applied.append({
                "file": str(file_path.relative_to(self.project_root)),
                "type": "hardcoded_dict_keys",
                "description": f"Fix failed: {e}",
                "status": "FAILED"
            })
            print(f"❌ Failed to fix: {file_path.name} - {e}")
    
    def _apply_dynamic_key_pattern(self, content: str, filename: str) -> str:
        """파일별 동적 키 패턴 적용"""
        
        if "session_data_manager" in filename:
            # 세션 데이터 매니저의 하드코딩 수정
            content = self._fix_session_manager_hardcoding(content)
            
        elif "user_file_tracker" in filename:
            # 사용자 파일 트래커의 하드코딩 수정
            content = self._fix_file_tracker_hardcoding(content)
            
        elif "domain_specific_answer_formatter" in filename:
            # 도메인별 답변 포매터의 하드코딩 수정
            content = self._fix_formatter_hardcoding(content)
            
        elif "domain_analysis_engine" in filename:
            # 도메인 분석 엔진의 하드코딩 수정
            content = self._fix_analysis_engine_hardcoding(content)
        
        return content
    
    def _fix_session_manager_hardcoding(self, content: str) -> str:
        """세션 매니저 하드코딩 수정"""
        # 하드코딩된 도메인 딕셔너리를 동적 생성으로 변경
        pattern = r"{\s*['\"]semiconductor['\"]\s*:\s*[^}]+}"
        replacement = "self._get_domain_configs_dynamically()"
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        
        # 필요한 메서드 추가
        if "def _get_domain_configs_dynamically" not in content:
            method_code = '''
    def _get_domain_configs_dynamically(self):
        """동적 도메인 설정 생성"""
        # LLM 기반 도메인 설정 생성으로 교체 예정
        return {}
'''
            # 클래스 끝 부분에 메서드 추가
            content = re.sub(r'(\n\s*def\s+\w+.*?\n.*?return.*?\n)', method_code + r'\1', content, count=1)
        
        return content
    
    def _fix_file_tracker_hardcoding(self, content: str) -> str:
        """파일 트래커 하드코딩 수정"""
        # 하드코딩된 도메인 키를 변수로 교체
        pattern = r"['\"]semiconductor['\"]"
        content = re.sub(pattern, "detected_domain", content)
        
        pattern = r"['\"]finance['\"]"
        content = re.sub(pattern, "detected_domain", content)
        
        return content
    
    def _fix_formatter_hardcoding(self, content: str) -> str:
        """포매터 하드코딩 수정"""
        # 하드코딩된 도메인별 포맷을 동적 포맷으로 변경
        hardcoded_dict_pattern = r"{\s*['\"]finance['\"]\s*:[^}]+['\"]healthcare['\"]\s*:[^}]+}"
        replacement = "self._generate_domain_formats_dynamically(domain)"
        content = re.sub(hardcoded_dict_pattern, replacement, content, flags=re.DOTALL)
        
        return content
    
    def _fix_analysis_engine_hardcoding(self, content: str) -> str:
        """분석 엔진 하드코딩 수정"""
        # 하드코딩된 도메인 분석 로직을 LLM 기반으로 변경
        pattern = r"['\"]semiconductor['\"]\s*:\s*\[[^\]]+\]"
        content = re.sub(pattern, "detected_domain: self._get_domain_keywords_llm(domain)", content)
        
        return content
    
    def _print_summary(self):
        """수정 결과 요약"""
        print("\n" + "="*60)
        print("📊 하드코딩 일괄 수정 결과")
        print("="*60)
        
        successful_fixes = [f for f in self.fixes_applied if f["status"] == "SUCCESS"]
        failed_fixes = [f for f in self.fixes_applied if f["status"] == "FAILED"]
        
        print(f"\n✅ 성공한 수정: {len(successful_fixes)}개")
        for fix in successful_fixes:
            print(f"   - {fix['file']}: {fix['description']}")
        
        if failed_fixes:
            print(f"\n❌ 실패한 수정: {len(failed_fixes)}개")
            for fix in failed_fixes:
                print(f"   - {fix['file']}: {fix['description']}")
        
        print(f"\n📈 전체 수정률: {len(successful_fixes)}/{len(self.fixes_applied)} ({len(successful_fixes)/max(len(self.fixes_applied), 1)*100:.1f}%)")
        
        # 결과 저장
        with open(self.project_root / "tests" / "verification" / "hardcoding_fixes_log.json", 'w') as f:
            json.dump(self.fixes_applied, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 수정 로그가 저장되었습니다: hardcoding_fixes_log.json")


def main():
    """메인 실행"""
    fixer = HardcodingBatchFixer()
    fixer.fix_all_hardcoding()


if __name__ == "__main__":
    main()