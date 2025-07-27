#!/usr/bin/env python3
"""Simple screenshot test to see current Streamlit interface"""

from playwright.sync_api import sync_playwright
import time

def take_screenshot():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(viewport={'width': 1920, 'height': 1080})
        page = context.new_page()
        
        try:
            print("Navigating to Streamlit app...")
            page.goto("http://localhost:8501", wait_until="domcontentloaded")
            
            # Wait for app to load
            page.wait_for_timeout(5000)
            
            # Take initial screenshot
            page.screenshot(path="tests/e2e/screenshots/streamlit_initial.png")
            print("Initial screenshot saved")
            
            # Upload file
            upload_input = page.locator("input[type='file']").first
            upload_input.set_input_files("/Users/gukil/CherryAI/CherryAI_0717/ion_implant_3lot_dataset.csv")
            print("File uploaded")
            
            # Wait and take another screenshot
            page.wait_for_timeout(3000)
            page.screenshot(path="tests/e2e/screenshots/streamlit_after_upload.png")
            print("After upload screenshot saved")
            
            # Look for all input elements
            inputs = page.locator("input").all()
            print(f"Found {len(inputs)} input elements:")
            for i, inp in enumerate(inputs):
                try:
                    placeholder = inp.get_attribute("placeholder")
                    input_type = inp.get_attribute("type")
                    class_name = inp.get_attribute("class")
                    print(f"  {i}: type={input_type}, placeholder='{placeholder}', class='{class_name}'")
                except:
                    print(f"  {i}: (unable to get attributes)")
            
            # Look for text areas too
            textareas = page.locator("textarea").all()
            print(f"Found {len(textareas)} textarea elements:")
            for i, ta in enumerate(textareas):
                try:
                    placeholder = ta.get_attribute("placeholder")
                    class_name = ta.get_attribute("class")
                    print(f"  {i}: placeholder='{placeholder}', class='{class_name}'")
                except:
                    print(f"  {i}: (unable to get attributes)")
            
            # Take final screenshot
            page.screenshot(path="tests/e2e/screenshots/streamlit_final.png")
            print("Final screenshot saved")
            
        finally:
            context.close()
            browser.close()

if __name__ == "__main__":
    take_screenshot()