
import { test } from '@playwright/test';
import { expect } from '@playwright/test';

test('CherryAI_E2E_Test_2025-07-26', async ({ page, context }) => {
  
    // Navigate to URL
    await page.goto('http://localhost:8501');

    // Take screenshot
    await page.screenshot({ path: 'initial_load.png' });

    // Navigate to URL
    await page.goto('http://localhost:8501');

    // Take screenshot
    await page.screenshot({ path: 'working_app_load.png' });

    // Take screenshot
    await page.screenshot({ path: 'after_file_upload.png' });

    // Click element
    await page.click('button:has-text("Run Analysis")');

    // Take screenshot
    await page.screenshot({ path: 'after_analysis.png' });

    // Fill input field
    await page.fill('div[data-testid="stChatInput"] textarea', 'What's the shape of my data?');

    // Take screenshot
    await page.screenshot({ path: 'after_chat_message.png' });

    // Click element
    await page.click('div[data-baseweb="select"] div[role="button"]');

    // Click element
    await page.click('div:has-text("Basic Statistics")');

    // Fill input field
    await page.fill('div[data-testid="stChatInput"] textarea', 'Show me the columns');

    // Take screenshot
    await page.screenshot({ path: 'second_chat_message.png' });
});