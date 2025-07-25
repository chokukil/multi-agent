"""
Page Object Models for Cherry AI Streamlit Platform

This module provides page object models for UI testing with Playwright.
"""

from typing import List, Optional, Dict, Any
from playwright.async_api import Page, ElementHandle, Locator
import asyncio
import logging

logger = logging.getLogger(__name__)


class BasePage:
    """Base page class with common functionality."""
    
    def __init__(self, page: Page):
        self.page = page
    
    async def wait_for_element(self, selector: str, timeout: int = 30000) -> Locator:
        """Wait for element to be visible."""
        locator = self.page.locator(selector)
        await locator.wait_for(state="visible", timeout=timeout)
        return locator
    
    async def take_screenshot(self, name: str):
        """Take a screenshot with a specific name."""
        await self.page.screenshot(path=f"tests/e2e/screenshots/{name}.png")
    
    async def get_text(self, selector: str) -> str:
        """Get text content of an element."""
        element = await self.wait_for_element(selector)
        return await element.text_content()
    
    async def click_button(self, text: str):
        """Click a button by its text."""
        button = self.page.locator(f'button:has-text("{text}")')
        await button.click()
    
    async def fill_input(self, selector: str, text: str):
        """Fill an input field."""
        input_field = await self.wait_for_element(selector)
        await input_field.fill(text)


class FileUploadPage(BasePage):
    """Page object for file upload functionality."""
    
    # Selectors
    UPLOAD_AREA = '[data-testid="stFileUploader"]'
    UPLOAD_BUTTON = 'button[data-testid="stFileUploaderButton"]'
    UPLOADED_FILES = '[data-testid="stFileUploaderFile"]'
    DATA_CARDS = '[data-testid="dataCard"]'
    PROGRESS_BAR = '[role="progressbar"]'
    
    async def upload_file(self, file_path: str):
        """Upload a file through the file uploader."""
        file_input = self.page.locator('input[type="file"]')
        await file_input.set_input_files(file_path)
        
        # Wait for upload to complete
        await self.page.wait_for_selector(self.PROGRESS_BAR, state="hidden", timeout=60000)
    
    async def upload_multiple_files(self, file_paths: List[str]):
        """Upload multiple files."""
        file_input = self.page.locator('input[type="file"]')
        await file_input.set_input_files(file_paths)
        
        # Wait for all uploads to complete
        await self.page.wait_for_selector(self.PROGRESS_BAR, state="hidden", timeout=60000)
    
    async def get_uploaded_file_names(self) -> List[str]:
        """Get list of uploaded file names."""
        files = await self.page.locator(self.UPLOADED_FILES).all()
        file_names = []
        for file in files:
            name = await file.text_content()
            file_names.append(name)
        return file_names
    
    async def get_data_cards(self) -> List[Dict[str, Any]]:
        """Get information from data cards."""
        cards = await self.page.locator(self.DATA_CARDS).all()
        data_cards = []
        
        for card in cards:
            card_info = {
                "name": await card.locator('[data-testid="cardName"]').text_content(),
                "rows": await card.locator('[data-testid="cardRows"]').text_content(),
                "columns": await card.locator('[data-testid="cardColumns"]').text_content(),
                "size": await card.locator('[data-testid="cardSize"]').text_content(),
                "selected": await card.locator('input[type="checkbox"]').is_checked()
            }
            data_cards.append(card_info)
        
        return data_cards
    
    async def select_data_card(self, card_name: str):
        """Select a data card by name."""
        card = self.page.locator(f'{self.DATA_CARDS}:has-text("{card_name}")')
        checkbox = card.locator('input[type="checkbox"]')
        if not await checkbox.is_checked():
            await checkbox.click()
    
    async def preview_dataset(self, card_name: str):
        """Click preview button for a dataset."""
        card = self.page.locator(f'{self.DATA_CARDS}:has-text("{card_name}")')
        preview_button = card.locator('button:has-text("Preview")')
        await preview_button.click()
        
        # Wait for preview modal
        await self.page.wait_for_selector('[data-testid="previewModal"]', timeout=5000)


class ChatInterfacePage(BasePage):
    """Page object for chat interface functionality."""
    
    # Selectors
    CHAT_INPUT = 'textarea[data-testid="chatInput"]'
    SEND_BUTTON = 'button[data-testid="sendButton"]'
    ATTACH_BUTTON = 'button[data-testid="attachButton"]'
    MESSAGE_CONTAINER = '[data-testid="messageContainer"]'
    USER_MESSAGE = '[data-testid="userMessage"]'
    AI_MESSAGE = '[data-testid="aiMessage"]'
    TYPING_INDICATOR = '[data-testid="typingIndicator"]'
    
    async def send_message(self, message: str):
        """Send a message through the chat interface."""
        # Fill the input
        await self.fill_input(self.CHAT_INPUT, message)
        
        # Click send button
        await self.page.locator(self.SEND_BUTTON).click()
        
        # Wait for typing indicator to appear and disappear
        await self.page.wait_for_selector(self.TYPING_INDICATOR, state="visible", timeout=5000)
        await self.page.wait_for_selector(self.TYPING_INDICATOR, state="hidden", timeout=120000)
    
    async def send_message_with_enter(self, message: str):
        """Send a message using Enter key."""
        await self.fill_input(self.CHAT_INPUT, message)
        await self.page.keyboard.press("Enter")
        
        # Wait for response
        await self.page.wait_for_selector(self.TYPING_INDICATOR, state="visible", timeout=5000)
        await self.page.wait_for_selector(self.TYPING_INDICATOR, state="hidden", timeout=120000)
    
    async def add_line_break(self, message: str):
        """Add a line break in message using Shift+Enter."""
        await self.fill_input(self.CHAT_INPUT, message)
        await self.page.keyboard.down("Shift")
        await self.page.keyboard.press("Enter")
        await self.page.keyboard.up("Shift")
    
    async def get_chat_messages(self) -> List[Dict[str, str]]:
        """Get all chat messages."""
        messages = []
        
        # Get user messages
        user_messages = await self.page.locator(self.USER_MESSAGE).all()
        for msg in user_messages:
            messages.append({
                "type": "user",
                "content": await msg.text_content()
            })
        
        # Get AI messages
        ai_messages = await self.page.locator(self.AI_MESSAGE).all()
        for msg in ai_messages:
            messages.append({
                "type": "ai",
                "content": await msg.text_content()
            })
        
        return messages
    
    async def wait_for_ai_response(self, timeout: int = 120000):
        """Wait for AI response to complete."""
        await self.page.wait_for_selector(self.TYPING_INDICATOR, state="hidden", timeout=timeout)
        
        # Wait a bit more to ensure response is fully rendered
        await asyncio.sleep(1)
    
    async def attach_file(self):
        """Click the attach button."""
        await self.page.locator(self.ATTACH_BUTTON).click()


class AgentCollaborationPage(BasePage):
    """Page object for agent collaboration visualization."""
    
    # Selectors
    AGENT_PROGRESS_CONTAINER = '[data-testid="agentProgressContainer"]'
    AGENT_CARD = '[data-testid="agentCard"]'
    PROGRESS_BAR = '[data-testid="agentProgressBar"]'
    AGENT_STATUS = '[data-testid="agentStatus"]'
    
    async def get_active_agents(self) -> List[Dict[str, Any]]:
        """Get information about active agents."""
        agents = []
        agent_cards = await self.page.locator(self.AGENT_CARD).all()
        
        for card in agent_cards:
            agent_info = {
                "name": await card.locator('[data-testid="agentName"]').text_content(),
                "status": await card.locator(self.AGENT_STATUS).text_content(),
                "progress": await card.locator(self.PROGRESS_BAR).get_attribute("aria-valuenow"),
                "task": await card.locator('[data-testid="agentTask"]').text_content()
            }
            agents.append(agent_info)
        
        return agents
    
    async def wait_for_agent_completion(self, agent_name: str, timeout: int = 120000):
        """Wait for a specific agent to complete."""
        agent_card = self.page.locator(f'{self.AGENT_CARD}:has-text("{agent_name}")')
        status = agent_card.locator(self.AGENT_STATUS)
        
        # Wait for status to be "completed"
        await status.wait_for(lambda: status.text_content() == "completed", timeout=timeout)
    
    async def get_agent_execution_time(self, agent_name: str) -> str:
        """Get execution time for a specific agent."""
        agent_card = self.page.locator(f'{self.AGENT_CARD}:has-text("{agent_name}")')
        exec_time = await agent_card.locator('[data-testid="executionTime"]').text_content()
        return exec_time


class ArtifactRendererPage(BasePage):
    """Page object for artifact rendering."""
    
    # Selectors
    ARTIFACT_CONTAINER = '[data-testid="artifactContainer"]'
    CHART_ARTIFACT = '[data-testid="chartArtifact"]'
    TABLE_ARTIFACT = '[data-testid="tableArtifact"]'
    CODE_ARTIFACT = '[data-testid="codeArtifact"]'
    DOWNLOAD_BUTTON = 'button[data-testid="downloadButton"]'
    EXPAND_BUTTON = 'button:has-text("View All Details")'
    
    async def get_artifacts(self) -> List[Dict[str, Any]]:
        """Get all rendered artifacts."""
        artifacts = []
        
        # Charts
        charts = await self.page.locator(self.CHART_ARTIFACT).all()
        for chart in charts:
            artifacts.append({
                "type": "chart",
                "title": await chart.locator('[data-testid="chartTitle"]').text_content()
            })
        
        # Tables
        tables = await self.page.locator(self.TABLE_ARTIFACT).all()
        for table in tables:
            artifacts.append({
                "type": "table",
                "title": await table.locator('[data-testid="tableTitle"]').text_content()
            })
        
        # Code blocks
        code_blocks = await self.page.locator(self.CODE_ARTIFACT).all()
        for code in code_blocks:
            artifacts.append({
                "type": "code",
                "language": await code.locator('[data-testid="codeLanguage"]').text_content()
            })
        
        return artifacts
    
    async def download_artifact(self, artifact_index: int, format: str = "raw"):
        """Download an artifact."""
        artifact = (await self.page.locator(self.ARTIFACT_CONTAINER).all())[artifact_index]
        download_btn = artifact.locator(f'{self.DOWNLOAD_BUTTON}:has-text("{format}")')
        
        # Start download
        async with self.page.expect_download() as download_info:
            await download_btn.click()
        
        download = await download_info.value
        return download
    
    async def expand_details(self):
        """Click expand details button."""
        await self.page.locator(self.EXPAND_BUTTON).click()
        
        # Wait for expansion animation
        await asyncio.sleep(0.5)
    
    async def copy_code(self, code_index: int):
        """Copy code from a code artifact."""
        code_blocks = await self.page.locator(self.CODE_ARTIFACT).all()
        code_block = code_blocks[code_index]
        copy_btn = code_block.locator('button:has-text("Copy")')
        await copy_btn.click()


class RecommendationPage(BasePage):
    """Page object for recommendation system."""
    
    # Selectors
    RECOMMENDATION_CARD = '[data-testid="recommendationCard"]'
    EXECUTE_BUTTON = 'button[data-testid="executeRecommendation"]'
    
    async def get_recommendations(self) -> List[Dict[str, Any]]:
        """Get all recommendations."""
        recommendations = []
        cards = await self.page.locator(self.RECOMMENDATION_CARD).all()
        
        for card in cards:
            rec_info = {
                "title": await card.locator('[data-testid="recTitle"]').text_content(),
                "description": await card.locator('[data-testid="recDescription"]').text_content(),
                "time_estimate": await card.locator('[data-testid="recTimeEstimate"]').text_content(),
                "complexity": await card.locator('[data-testid="recComplexity"]').text_content()
            }
            recommendations.append(rec_info)
        
        return recommendations
    
    async def execute_recommendation(self, title: str):
        """Execute a recommendation by title."""
        card = self.page.locator(f'{self.RECOMMENDATION_CARD}:has-text("{title}")')
        execute_btn = card.locator(self.EXECUTE_BUTTON)
        await execute_btn.click()
        
        # Wait for execution to start
        await self.page.wait_for_selector('[data-testid="executionProgress"]', timeout=5000)


class CherryAIApp:
    """Main application page object aggregating all components."""
    
    def __init__(self, page: Page):
        self.page = page
        self.file_upload = FileUploadPage(page)
        self.chat = ChatInterfacePage(page)
        self.agents = AgentCollaborationPage(page)
        self.artifacts = ArtifactRendererPage(page)
        self.recommendations = RecommendationPage(page)
    
    async def wait_for_app_ready(self):
        """Wait for the application to be fully loaded."""
        # Wait for main app container
        await self.page.wait_for_selector('[data-testid="stApp"]', timeout=30000)
        
        # Wait for Cherry AI specific elements
        await self.page.wait_for_selector('text=Cherry AI Data Science Platform', timeout=10000)
        
        # Wait for chat interface to be ready
        await self.page.wait_for_selector(self.chat.CHAT_INPUT, timeout=10000)
        
        logger.info("Cherry AI app is ready")
    
    async def perform_analysis(self, file_path: str, query: str) -> Dict[str, Any]:
        """Perform a complete analysis workflow."""
        # Upload file
        await self.file_upload.upload_file(file_path)
        
        # Get data cards
        data_cards = await self.file_upload.get_data_cards()
        
        # Send analysis query
        await self.chat.send_message(query)
        
        # Wait for response
        await self.chat.wait_for_ai_response()
        
        # Get artifacts
        artifacts = await self.artifacts.get_artifacts()
        
        # Get recommendations
        recommendations = await self.recommendations.get_recommendations()
        
        return {
            "data_cards": data_cards,
            "artifacts": artifacts,
            "recommendations": recommendations
        }