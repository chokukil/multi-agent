#!/usr/bin/env python3
"""
macOS matplotlib 한글 폰트 설정 스크립트
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import platform

def find_korean_fonts():
    """시스템에서 사용 가능한 한글 폰트 찾기"""
    print("🔍 시스템에서 사용 가능한 한글 폰트 검색 중...")
    
    # 한글 폰트 후보들
    korean_font_candidates = [
        'AppleGothic',
        'Apple SD Gothic Neo',
        'NanumGothic',
        'NanumBarunGothic',
        'Noto Sans CJK KR',
        'Noto Sans KR',
        'Malgun Gothic',
        'Gulim',
        'Dotum',
        'Batang'
    ]
    
    available_fonts = []
    all_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font_name in korean_font_candidates:
        if font_name in all_fonts:
            available_fonts.append(font_name)
            print(f"✅ {font_name}: 사용 가능")
        else:
            print(f"❌ {font_name}: 사용 불가")
    
    print(f"\n📊 총 {len(available_fonts)}개의 한글 폰트 발견: {available_fonts}")
    return available_fonts

def set_korean_font():
    """최적의 한글 폰트 설정"""
    print("\n🔧 matplotlib 한글 폰트 설정 중...")
    
    available_fonts = find_korean_fonts()
    
    if not available_fonts:
        print("❌ 사용 가능한 한글 폰트를 찾을 수 없습니다.")
        print("🔍 시스템 기본 폰트로 fallback 합니다.")
        
        # macOS 기본 폰트 사용
        if platform.system() == 'Darwin':  # macOS
            plt.rcParams['font.family'] = ['Arial Unicode MS', 'AppleGothic']
        else:
            plt.rcParams['font.family'] = ['DejaVu Sans']
        
        return None
    
    # 우선순위대로 폰트 설정 시도
    preferred_fonts = [
        'Apple SD Gothic Neo',
        'AppleGothic', 
        'NanumGothic',
        'NanumBarunGothic'
    ]
    
    selected_font = None
    for font in preferred_fonts:
        if font in available_fonts:
            selected_font = font
            break
    
    if not selected_font:
        selected_font = available_fonts[0]
    
    print(f"✅ 선택된 폰트: {selected_font}")
    
    # matplotlib 설정
    plt.rcParams['font.family'] = selected_font
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
    
    return selected_font

def clear_font_cache():
    """matplotlib 폰트 캐시 초기화"""
    print("🔄 matplotlib 폰트 캐시 초기화 중...")
    
    try:
        # 폰트 캐시 파일 위치 찾기
        cache_dir = fm.get_cachedir()
        cache_file = os.path.join(cache_dir, 'fontlist-v330.json')
        
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print(f"✅ 폰트 캐시 파일 삭제: {cache_file}")
        
        # 폰트 매니저 재빌드
        fm._rebuild()
        print("✅ 폰트 매니저 재빌드 완료")
        
    except Exception as e:
        print(f"⚠️ 폰트 캐시 초기화 중 오류: {e}")

def test_korean_display():
    """한글 표시 테스트"""
    print("\n🧪 한글 폰트 표시 테스트 중...")
    
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 테스트 데이터
        categories = ['카테고리A', '카테고리B', '카테고리C', '카테고리D']
        values = [23, 45, 56, 78]
        
        ax.bar(categories, values)
        ax.set_title('한글 폰트 테스트 - 막대그래프', fontsize=16)
        ax.set_xlabel('카테고리', fontsize=12)
        ax.set_ylabel('값', fontsize=12)
        
        # 그래프를 파일로 저장 (표시하지 않음)
        plt.savefig('korean_font_test.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ 한글 폰트 테스트 완료 - korean_font_test.png 저장됨")
        return True
        
    except Exception as e:
        print(f"❌ 한글 폰트 테스트 실패: {e}")
        return False

def get_current_font_settings():
    """현재 matplotlib 폰트 설정 확인"""
    print("\n📋 현재 matplotlib 폰트 설정:")
    print(f"   font.family: {plt.rcParams['font.family']}")
    print(f"   font.size: {plt.rcParams['font.size']}")
    print(f"   axes.unicode_minus: {plt.rcParams['axes.unicode_minus']}")

def main():
    """메인 실행 함수"""
    print("🔧 macOS matplotlib 한글 폰트 설정 도구")
    print("=" * 50)
    
    print(f"🖥️ 운영체제: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {platform.python_version()}")
    
    try:
        import matplotlib
        print(f"📊 matplotlib: {matplotlib.__version__}")
    except ImportError:
        print("❌ matplotlib이 설치되지 않았습니다.")
        return
    
    # 1. 현재 설정 확인
    get_current_font_settings()
    
    # 2. 폰트 캐시 초기화
    clear_font_cache()
    
    # 3. 한글 폰트 찾기 및 설정
    selected_font = set_korean_font()
    
    # 4. 새로운 설정 확인
    get_current_font_settings()
    
    # 5. 한글 표시 테스트
    if test_korean_display():
        print("\n🎉 한글 폰트 설정이 성공적으로 완료되었습니다!")
        
        if selected_font:
            print(f"📝 설정된 폰트: {selected_font}")
            print("\n📌 사용 방법:")
            print("   다음 코드를 Python 스크립트 상단에 추가하세요:")
            print(f"""
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = '{selected_font}'
plt.rcParams['axes.unicode_minus'] = False
""")
        
        print("\n🔧 Enhanced EDA v2 서버에 자동 적용됩니다.")
    else:
        print("\n⚠️ 한글 폰트 설정에 문제가 있습니다.")
        print("   시스템 폰트를 확인하거나 한글 폰트를 설치해주세요.")

if __name__ == "__main__":
    main() 