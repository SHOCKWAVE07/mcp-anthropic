from typing import List, Optional
from mcp_client import MCPClient
from core.tools import ToolManager
from core.gemini import Gemini
import google.genai as genai


class Chat:
    def __init__(self, gemini_service: Gemini, clients: dict[str, MCPClient]):
        self.gemini_service: Gemini = gemini_service
        self.clients: dict[str, MCPClient] = clients
        self.messages: list[genai.types.MessageParam] = []

    async def _process_query(self, query: str):
        self.messages.append({"role": "user", "parts": query})

    async def run(self, query: str) -> str:
        final_text_response = ""

        await self._process_query(query)

        while True:
            response = self.gemini_service.chat(
                messages=self.messages,
                tool_config=await ToolManager.get_all_tools(self.clients),
            )

            self.gemini_service.add_assistant_message(self.messages, response)

            # Check if a tool was used in the response
            tool_use_occurred = False
            for part in response.parts:
                if part.function_call:
                    tool_use_occurred = True
                    break

            if tool_use_occurred:
                # Assuming text_from_message handles empty responses correctly
                print(self.gemini_service.text_from_message(response))
                
                tool_result_parts = await ToolManager.execute_tool_requests(
                    self.clients, response
                )

                self.gemini_service.add_user_message(
                    self.messages, tool_result_parts
                )
            else:
                final_text_response = self.gemini_service.text_from_message(response)
                break

        return final_text_response