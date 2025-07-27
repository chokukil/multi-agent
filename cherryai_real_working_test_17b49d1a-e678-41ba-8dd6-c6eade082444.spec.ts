
import { test } from '@playwright/test';
import { expect } from '@playwright/test';

test('CherryAI_Real_Working_Test_2025-07-26', async ({ page, context }) => {
  
    // Navigate to URL
    await page.goto('http://localhost:8501');

    // Take screenshot
    await page.screenshot({ path: 'real_working_app.png' });

    // Take screenshot
    await page.screenshot({ path: 'after_real_upload.png' });

    // Fill input field
    await page.fill('div[data-testid="stChatInput"] textarea', 'What can you tell me about this data?');

    // Take screenshot
    await page.screenshot({ path: 'after_chat_question.png' });

    // Click element
    await page.click('button:has-text("📊 Get Summary")');

    // Navigate to URL
    await page.goto('http://localhost:8501');

    // Navigate to URL
    await page.goto('http://localhost:8501');

    // Navigate to URL
    await page.goto('http://localhost:8501');

    // Take screenshot
    await page.screenshot({ path: 'stable_app_final.png' });

    // Take screenshot
    await page.screenshot({ path: 'stable_after_upload.png' });

    // Click element
    await page.click('button:has-text("Get Summary")');

    // Take screenshot
    await page.screenshot({ path: 'stable_after_summary.png' });

    // Fill input field
    await page.fill('div[data-testid="stChatInput"] textarea', 'What are the columns in this data?');

    // Take screenshot
    await page.screenshot({ path: 'stable_final_test.png' });
});