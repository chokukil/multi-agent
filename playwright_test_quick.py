#!/usr/bin/env python3
"""
간단한 Playwright 테스트 - CherryAI 시스템 확인
"""

import asyncio
from playwright.async_api import async_playwright

async def test_cherry_ai():
    print("🎭 Playwright CherryAI 테스트 시작...")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        try:
            print('📍 CherryAI 페이지 접속 중...')
            await page.goto('http://localhost:8501', timeout=30000)
            
            print('⏳ 페이지 로딩 대기 중...')
            await page.wait_for_timeout(5000)
            
            title = await page.title()
            print(f'📄 페이지 제목: {title}')
            
            # Streamlit 앱 요소 확인
            streamlit_elements = await page.query_selector_all('[data-testid*="st"]')
            print(f'🔍 Streamlit 요소: {len(streamlit_elements)}개 발견')
            
            # 파일 업로더 확인
            file_uploaders = await page.query_selector_all('input[type="file"]')
            print(f'📤 파일 업로더: {len(file_uploaders)}개 발견')
            
            # 텍스트 입력 요소 확인
            text_inputs = await page.query_selector_all('textarea, input[type="text"]')
            print(f'💬 텍스트 입력: {len(text_inputs)}개 발견')
            
            # 기본 텍스트 확인
            content = await page.content()
            if 'CherryAI' in content or 'Cherry AI' in content:
                print('✅ CherryAI 콘텐츠 확인')
            else:
                print('⚠️ CherryAI 콘텐츠 미확인')
            
            # 스크린샷 촬영
            await page.screenshot(path='playwright_test_screenshot.png', full_page=True)
            print('📸 스크린샷 저장: playwright_test_screenshot.png')
            
            print('🎉 Playwright 테스트 완료!')
            return True
            
        except Exception as e:
            print(f'❌ 테스트 실패: {e}')
            return False
            
        finally:
            await browser.close()

if __name__ == "__main__":
    success = asyncio.run(test_cherry_ai())
    exit(0 if success else 1) 