
import { test } from '@playwright/test';
import { expect } from '@playwright/test';

test('DataManagerA2ATest_2025-06-28', async ({ page, context }) => {
  
    // Navigate to URL
    await page.goto('http://localhost:8501');

    // Take screenshot
    await page.screenshot({ path: 'cherryai_main_interface.png', { fullPage: true } });

    // Fill input field
    await page.fill('textarea[data-testid="stTextArea"]', '반도체 이온주입 공정 전문가로서 ion_implant_3lot_dataset.xlsx 데이터로 EDA 진행해줘');

    // Click element
    await page.click('input[type="file"]');

    // Fill input field
    await page.fill('textarea[data-testid="stChatInputTextArea"]', '반도체 이온주입 공정 전문가로서 EDA 진행해줘');

    // Take screenshot
    await page.screenshot({ path: 'a2a_request_sent.png', { fullPage: true } });

    // Take screenshot
    await page.screenshot({ path: 'a2a_response_received.png', { fullPage: true } });
});