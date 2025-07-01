
import { test } from '@playwright/test';
import { expect } from '@playwright/test';

test('CherryAI_FinalReport_Test_2025-07-01', async ({ page, context }) => {
  
    // Navigate to URL
    await page.goto('http://localhost:8501');

    // Take screenshot
    await page.screenshot({ path: 'initial_page.png', { fullPage: true } });

    // Fill input field
    await page.fill('textarea[aria-label='Chat input']', '이 반도체 이온 주입 데이터를 분석해서 종합적인 분석 보고서를 만들어주세요. 특히 TW AVG 값의 패턴과 장비별 성능을 중심으로 분석해주세요.');

    // Fill input field
    await page.fill('textarea[data-testid='stChatInputTextArea']', '이 반도체 이온 주입 데이터를 분석해서 종합적인 분석 보고서를 만들어주세요. 특히 TW AVG 값의 패턴과 장비별 성능을 중심으로 분석해주세요.');

    // Click element
    await page.click('button[data-testid='stChatInputSubmitButton']');

    // Take screenshot
    await page.screenshot({ path: 'analysis_running.png', { fullPage: true } });

    // Take screenshot
    await page.screenshot({ path: 'analysis_completed.png', { fullPage: true } });

    // Take screenshot
    await page.screenshot({ path: 'final_analysis_result.png', { fullPage: true } });

    // Take screenshot
    await page.screenshot({ path: 'final_report_status.png', { fullPage: true } });
});