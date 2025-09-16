# from typing import List, Optional
# from mcp_client import MCPClient
# from core.tools import ToolManager
# from core.gemini import Gemini
# import google.genai as genai


# class Chat:
#     def __init__(self, gemini_service: Gemini, clients: dict[str, MCPClient]):
#         self.gemini_service: Gemini = gemini_service
#         self.clients: dict[str, MCPClient] = clients
#         self.messages: list[genai.types.MessageParam] = []

#     async def _process_query(self, query: str):
#         self.messages.append({"role": "user", "parts": query})

#     async def run(self, query: str) -> str:
#         final_text_response = ""

#         await self._process_query(query)

#         while True:
#             response = self.gemini_service.chat(
#                 messages=self.messages,
#                 tool_config=await ToolManager.get_all_tools(self.clients),
#             )

#             self.gemini_service.add_assistant_message(self.messages, response)

#             # Check if a tool was used in the response
#             tool_use_occurred = False
#             for part in response.parts:
#                 if part.function_call:
#                     tool_use_occurred = True
#                     break

#             if tool_use_occurred:
#                 # Add a descriptive print statement when a tool is called
#                 text_response = self.gemini_service.text_from_message(response)
#                 if text_response:
#                     print(text_response)
#                 print("Calling a tool...") 
                
#                 tool_result_parts = await ToolManager.execute_tool_requests(
#                     self.clients, response
#                 )

#                 self.gemini_service.add_user_message(
#                     self.messages, tool_result_parts
#                 )
#             else:
#                 final_text_response = self.gemini_service.text_from_message(response)
#                 if not final_text_response:
#                     return "The model did not provide a text response."
#                 break

#         return final_text_response

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
        self.messages.append({"role": "user", "parts": [query]})

    async def run(self, query: str) -> str:
        final_text_response = ""

        # Use the query to determine if tools should be enabled
        use_tools = query.startswith("/") or "@" in query

        await self._process_query(query)

        while True:
            # Conditionally pass tool_config based on the query
            if use_tools:
                tool_config = await ToolManager.get_all_tools(self.clients)
            else:
                tool_config = None
                
            response = self.gemini_service.chat(
                messages=self.messages,
                tool_config=tool_config,
            )

            self.gemini_service.add_assistant_message(self.messages, response)

            tool_use_occurred = False
            for part in response.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    tool_use_occurred = True
                    break

            if tool_use_occurred:
                text_response = self.gemini_service.text_from_message(response)
                if text_response:
                    print(text_response)
                print("Calling a tool...") 
                
                tool_result_parts = await ToolManager.execute_tool_requests(
                    self.clients, response
                )

                self.gemini_service.add_user_message(
                    self.messages, tool_result_parts
                )
            else:
                final_text_response = self.gemini_service.text_from_message(response)
                if not final_text_response:
                    return "The model did not provide a text response."
                break

        return final_text_response