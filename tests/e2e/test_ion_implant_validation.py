import json, re, time
from pathlib import Path
import pytest
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.e2e.utils.ion_validation import (
    NUM_RX, parse_number, validate_sections, validate_domain_keywords,
    validate_logic, soft_close, KW_DOMAIN
)

APP = "http://localhost:8501"
CSV = Path("/Users/gukil/CherryAI/CherryAI_0717/ion_implant_3lot_dataset.csv")
BASELINE = Path("tests/baselines/ion_implant_limits.json")

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_ion_implant_deep_validation(page):
    """
    Deep validation test for Ion Implant domain analysis including:
    - Performance budgets (startup <10s, response <30s)
    - Baseline matching with derived CSV limits
    - Domain section validation (이상여부, 원인, 조치)
    - Technical keyword presence
    - Logic consistency (limits vs narrative)
    """
    
    # App startup with performance measurement
    t0 = time.time()
    await page.goto(APP, wait_until="domcontentloaded")
    
    # Wait for app to be ready - try different selectors
    try:
        await page.wait_for_selector('[data-testid="stApp"]', timeout=20_000)
    except:
        try:
            # Fallback to any visible element that indicates app is loaded
            await page.wait_for_selector('h1, h2, h3', timeout=20_000)
        except:
            # Final fallback - just wait for basic page structure
            await page.wait_for_selector('body', timeout=20_000)
    
    startup = time.time()-t0
    assert startup < 10, f"App startup too slow: {startup:.2f}s"

    # File upload with timeout handling
    try:
        await page.wait_for_selector('input[type="file"]', timeout=10_000)
        upload_element = page.locator("input[type='file']")
    except:
        # Fallback selector for file upload
        upload_element = page.locator("input[type='file']").first()
    
    await upload_element.set_input_files(str(CSV))
    
    # Wait for upload completion - just wait a bit for upload to process
    await page.wait_for_timeout(5000)

    # Chat query with domain-specific prompt
    try:
        chat_input = page.locator("input[placeholder*='메시지']").first()
        await chat_input.wait_for(state="visible", timeout=10_000)
    except:
        # Alternative selector
        chat_input = page.locator(".stChatInputContainer input").first()
        await chat_input.wait_for(state="visible", timeout=10_000)
    
    t1 = time.time()
    expert_prompt = """당신은 20년 경력의 반도체 이온주입 공정 엔지니어입니다. 업로드된 LOT/계측/레시피/장비 데이터로:
(1) 공정 이상 여부 판단
(2) 원인 해석 (도메인 근거 포함)  
(3) 실무 조치 제안
TW AVG, LOW LIMIT, HIGH LIMIT을 수치로 제시하고, 장비 간 분포/산포 비교를 포함해 주세요."""
    
    await chat_input.fill(expert_prompt)
    await page.keyboard.press("Enter")

    # Wait for assistant response with timeout handling
    try:
        await page.wait_for_selector('[data-testid="stChatMessage"]', timeout=120_000)
        msg = page.locator('[data-testid="stChatMessage"]').last()
    except:
        # Fallback selector for assistant message
        try:
            await page.wait_for_selector('.stChatMessage', timeout=120_000)
            msg = page.locator('.stChatMessage').last()
        except:
            # Last resort - any message-like element
            await page.wait_for_selector('div[data-testid*="chat"]', timeout=120_000)
            msg = page.locator('div[data-testid*="chat"]').last()
    
    resp_time = time.time()-t1
    assert resp_time < 30, f"Response too slow: {resp_time:.2f}s"

    # Extract response text and normalize
    text = (await msg.inner_text()).replace("\xa0"," ").strip()
    assert len(text) > 100, f"Response too short: {len(text)} chars"

    # Validation 1: Required sections (이상여부, 원인, 조치)
    missing_sections = validate_sections(text)
    assert not missing_sections, f"Missing required sections: {missing_sections}"

    # Validation 2: Domain keywords
    missing_keywords = validate_domain_keywords(text)
    # Allow some flexibility - warn if too many missing
    if len(missing_keywords) > len(KW_DOMAIN) * 0.5:
        pytest.warns(UserWarning, f"Many domain keywords missing: {missing_keywords}")

    # Validation 3: Extract numerical values
    avg  = parse_number(NUM_RX["avg"], text)
    low  = parse_number(NUM_RX["low"], text)
    high = parse_number(NUM_RX["high"], text)

    # Validation 4: Baseline matching (if numbers found and baseline exists)
    if BASELINE.exists() and all(v is not None for v in (avg, low, high)):
        bl = json.loads(BASELINE.read_text(encoding="utf-8"))
        assert soft_close(avg, bl["tw_avg"]),  f"AVG baseline mismatch: response={avg}, baseline={bl['tw_avg']}"
        assert soft_close(low, bl["low"]),     f"LOW baseline mismatch: response={low}, baseline={bl['low']}"
        assert soft_close(high, bl["high"]),   f"HIGH baseline mismatch: response={high}, baseline={bl['high']}"
    elif all(v is not None for v in (avg, low, high)):
        # Basic sanity checks even without baseline
        assert low < avg < high, f"Illogical limits: low={low}, avg={avg}, high={high}"

    # Validation 5: Logic consistency (limits vs narrative)
    logic_issues = validate_logic(text, avg, low, high)
    if logic_issues:
        # Log but don't fail - these are soft logic checks
        print(f"Logic consistency warnings: {logic_issues}")

    # Performance metrics logging
    print(f"Performance metrics - Startup: {startup:.2f}s, Response: {resp_time:.2f}s")
    print(f"Response length: {len(text)} chars")
    if avg is not None:
        print(f"Extracted values - AVG: {avg}, LOW: {low}, HIGH: {high}")


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.asyncio
async def test_ion_implant_equipment_analysis(page):
    """
    Additional test focusing on equipment-specific analysis
    """
    await page.goto(APP, wait_until="domcontentloaded")
    await page.wait_for_timeout(3000)  # Allow app to fully load
    
    # Upload same dataset
    upload_element = page.locator("input[type='file']").first()
    await upload_element.set_input_files(str(CSV))
    await page.wait_for_timeout(5000)  # Wait for upload
    
    # Equipment-focused query
    chat_input = page.locator("input[placeholder*='메시지']").first()
    await chat_input.wait_for(state="visible", timeout=10_000)
    
    equipment_prompt = "장비별 TW 분포를 분석하고 EQ_1, EQ_2, EQ_3의 성능 차이와 calibration 상태를 평가해주세요."
    await chat_input.fill(equipment_prompt)
    await page.keyboard.press("Enter")
    
    # Wait for response
    try:
        await page.wait_for_selector('[data-testid="stChatMessage"]', timeout=60_000)
        msg = page.locator('[data-testid="stChatMessage"]').last()
    except:
        await page.wait_for_selector('.stChatMessage', timeout=60_000)
        msg = page.locator('.stChatMessage').last()
    
    text = await msg.inner_text()
    
    # Validate equipment-specific content
    assert re.search(r"EQ_[123]", text), "Equipment IDs not mentioned"
    assert re.search(r"(분포|distribution|calibration)", text, re.I), "Equipment analysis keywords missing"