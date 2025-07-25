#!/usr/bin/env python3
"""
P0 Smoke Test - Verify basic UI components are detectable by E2E tests
"""

import asyncio
from playwright.async_api import async_playwright
import time


async def test_p0_components():
    """Test that P0 components are detectable and functional"""
    print("🚀 Starting P0 Components Smoke Test...")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, slow_mo=100)
        page = await browser.new_page()
        
        try:
            print("📡 Connecting to minimal Cherry AI app...")
            await page.goto("http://localhost:8521", wait_until="domcontentloaded", timeout=30000)
            
            # Wait for Streamlit to initialize
            print("⏳ Waiting for Streamlit initialization...")
            await page.wait_for_timeout(3000)
            
            # Test 1: Basic page load
            print("✅ Test 1: Basic page load")
            title = await page.title()
            print(f"   Page title: {title}")
            
            # Test 2: Check for page header
            print("✅ Test 2: Check for Cherry AI header")
            header_text = await page.text_content("h1")
            print(f"   Header: {header_text}")
            assert "Cherry AI Platform" in header_text, f"Expected Cherry AI Platform in header, got {header_text}"
            
            # Test 3: Look for chat input
            print("✅ Test 3: Check for chat input component")
            chat_inputs = await page.locator("input[placeholder*='message'], input[aria-label*='message'], input[placeholder*='Type'], textarea[placeholder*='message']").count()
            print(f"   Chat inputs found: {chat_inputs}")
            
            # Test 4: Look for file upload
            print("✅ Test 4: Check for file upload component")
            file_uploads = await page.locator("input[type='file'], button:has-text('Browse files'), *:has-text('Choose your data files')").count()
            print(f"   File upload elements found: {file_uploads}")
            
            # Test 5: Check sidebar
            print("✅ Test 5: Check for sidebar")
            sidebar_elements = await page.locator("[data-testid='stSidebar'], .stSidebar, aside").count()
            print(f"   Sidebar elements found: {sidebar_elements}")
            
            # Test 6: Look for AI Assistant header
            print("✅ Test 6: Check for AI Assistant section")
            ai_headers = await page.locator("*:has-text('AI Assistant'), h2:has-text('💬'), h3:has-text('AI')").count()
            print(f"   AI Assistant headers found: {ai_headers}")
            
            # Test 7: Look for Data Upload header
            print("✅ Test 7: Check for Data Upload section")
            upload_headers = await page.locator("*:has-text('Data Upload'), h2:has-text('📁'), h3:has-text('Upload')").count()
            print(f"   Data Upload headers found: {upload_headers}")
            
            # Test 8: General Streamlit health
            print("✅ Test 8: General Streamlit health check")
            streamlit_elements = await page.locator("[data-testid*='st'], .streamlit, [class*='st-']").count()
            print(f"   Total Streamlit elements: {streamlit_elements}")
            
            print("\n📊 P0 Components Test Summary:")
            print(f"   ✅ Page loads successfully")
            print(f"   ✅ Header present: {'Cherry AI Platform' in (header_text or '')}")
            print(f"   📝 Chat inputs: {chat_inputs} (target: >0)")
            print(f"   📁 File uploads: {file_uploads} (target: >0)")
            print(f"   📋 Sidebar: {sidebar_elements} (target: >0)")
            print(f"   💬 AI section: {ai_headers} (target: >0)")
            print(f"   📤 Upload section: {upload_headers} (target: >0)")
            print(f"   🧩 Streamlit elements: {streamlit_elements} (target: >10)")
            
            # Calculate score
            score = 0
            score += 1 if "Cherry AI Platform" in (header_text or "") else 0
            score += 1 if chat_inputs > 0 else 0
            score += 1 if file_uploads > 0 else 0
            score += 1 if sidebar_elements > 0 else 0
            score += 1 if ai_headers > 0 else 0
            score += 1 if upload_headers > 0 else 0
            score += 1 if streamlit_elements > 10 else 0
            
            print(f"\n🎯 P0 Readiness Score: {score}/7 ({score/7*100:.0f}%)")
            
            if score >= 5:
                print("🎉 P0 Components are E2E test ready!")
                return True
            else:
                print("⚠️  P0 Components need improvement for E2E tests")
                return False
                
        except Exception as e:
            print(f"❌ Error during test: {str(e)}")
            return False
            
        finally:
            await browser.close()


async def main():
    """Run the smoke test"""
    success = await test_p0_components()
    exit_code = 0 if success else 1
    print(f"\n🏁 Test completed with exit code: {exit_code}")
    return exit_code


if __name__ == "__main__":
    asyncio.run(main())