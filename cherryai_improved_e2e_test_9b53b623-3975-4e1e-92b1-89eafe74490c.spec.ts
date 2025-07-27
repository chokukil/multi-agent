
import { test } from '@playwright/test';
import { expect } from '@playwright/test';

test('CherryAI_Improved_E2E_Test_2025-07-26', async ({ page, context }) => {
  
    // Navigate to URL
    await page.goto('http://localhost:8501');

    // Take screenshot
    await page.screenshot({ path: 'improved_app_load.png' });

    // Take screenshot
    await page.screenshot({ path: 'improved_after_upload.png' });

    // Click element
    await page.click('input[value="Data Visualization"]');

    // Click element
    await page.click('label:has-text("Data Visualization")');

    // Click element
    await page.click('button:has-text("🚀 Run Analysis")');

    // Click element
    await page.click('button[key="run_analysis_btn"]');

    // Navigate to URL
    await page.goto('http://localhost:8501');

    // Take screenshot
    await page.screenshot({ path: 'fixed_app_load.png' });

    // Take screenshot
    await page.screenshot({ path: 'fixed_after_upload.png' });

    // Click element
    await page.click('label:has-text("Data Visualization")');

    // Click element
    await page.click('button:has-text("🚀 Run Analysis")');

    // Take screenshot
    await page.screenshot({ path: 'fixed_after_visualization.png' });

    // Fill input field
    await page.fill('div[data-testid="stChatInput"] textarea', 'What insights can you give me about this data?');

    // Take screenshot
    await page.screenshot({ path: 'fixed_final_test.png' });
});