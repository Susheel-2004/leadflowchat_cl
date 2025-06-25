import chainlit as cl
import requests
import os
from dotenv import load_dotenv
import json
from typing import List, Dict, Optional
import uuid
import hashlib
import time
import pickle
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('chainlit_demo.log')
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://your-api-base-url.com")
API_CHAT_ENDPOINT = f"{API_BASE_URL}/chat"
DEFAULT_MODEL = "deepseek-r1-distill-llama-70b"

# Cache configuration
CACHE_DURATION = 3600  # 1 hour in seconds
CACHE_FILE = Path("cache/api_cache.pkl")
CACHE_DIR = Path("cache")

# Ensure cache directory exists
CACHE_DIR.mkdir(exist_ok=True)

def load_cache_from_file() -> Dict:
    """Load cache from file if it exists"""
    try:
        if CACHE_FILE.exists():
            with open(CACHE_FILE, 'rb') as f:
                cache_data = pickle.load(f)
                logger.info(f"Loaded cache with {len(cache_data)} entries from file")
                return cache_data
    except Exception as e:
        logger.error(f"Error loading cache from file: {e}")
    
    logger.info("Starting with empty cache")
    return {}

def save_cache_to_file(cache_data: Dict):
    """Save cache to file"""
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(cache_data, f)
        logger.info(f"Saved cache with {len(cache_data)} entries to file")
        with open("test_reponse.json", "w") as test_file:
            json.dump(cache_data, test_file, indent=2)
    except Exception as e:
        logger.error(f"Error saving cache to file: {e}")

def cleanup_expired_cache(cache_data: Dict) -> Dict:
    """Remove expired entries from cache"""
    current_time = time.time()
    cleaned_cache = {}
    expired_count = 0
    
    for cache_key, cached_item in cache_data.items():
        if current_time - cached_item["timestamp"] < CACHE_DURATION:
            cleaned_cache[cache_key] = cached_item
        else:
            expired_count += 1
    
    if expired_count > 0:
        logger.info(f"Cleaned {expired_count} expired cache entries")
        save_cache_to_file(cleaned_cache)
    
    return cleaned_cache

# Load cache on startup
api_cache = load_cache_from_file()
api_cache = cleanup_expired_cache(api_cache)

def get_cache_key(messages: List[Dict], session_id: str, model: str) -> str:
    """Generate a cache key for the API request"""
    # Create a string from the request parameters
    cache_data = {
        "messages": messages,
        "model": model
        # Note: Not including session_id to allow cross-session caching
    }
    cache_string = json.dumps(cache_data, sort_keys=True)
    return hashlib.md5(cache_string.encode()).hexdigest()

def get_cached_response(cache_key: str) -> Optional[Dict]:
    """Get cached response if it exists and is not expired"""
    global api_cache
    
    if cache_key in api_cache:
        cached_data = api_cache[cache_key]
        if time.time() - cached_data["timestamp"] < CACHE_DURATION:
            logger.info(f"Cache HIT for key: {cache_key[:8]}...")
            return cached_data["response"]
        else:
            # Remove expired cache entry
            del api_cache[cache_key]
            save_cache_to_file(api_cache)
            logger.info(f"Cache EXPIRED for key: {cache_key[:8]}...")
    return None

def set_cached_response(cache_key: str, response: Dict):
    """Store response in cache and save to file"""
    global api_cache
    
    api_cache[cache_key] = {
        "response": response,
        "timestamp": time.time()
    }
    
    # Save to file after adding new entry
    save_cache_to_file(api_cache)
    logger.info(f"Cache SET for key: {cache_key[:8]}...")

class ConversationalChatClient:
    def __init__(self, api_base_url: str):
        self.api_base_url = api_base_url
        self.chat_endpoint = f"{api_base_url}/chat"
        self.models_endpoint = f"{api_base_url}/get_models"
        
    async def send_chat_message(self, messages: List[Dict], session_id: Optional[str] = None, model: str = DEFAULT_MODEL) -> Dict:
        """Send chat message to the conversational API"""
        try:
            payload = {
                "messages": messages,
                "session_id": session_id,
                "model": model
            }
            
            headers = {
                'Content-Type': 'application/json'
            }
            
            logger.info(f"Making request to: {self.chat_endpoint}")
            logger.debug(f"Payload: {json.dumps(payload, indent=2)}")
            
            # Check cache first
            cache_key = get_cache_key(messages, session_id, model)
            cached_response = get_cached_response(cache_key)
            if cached_response is not None:
                return cached_response
            
            response = requests.post(
                self.chat_endpoint,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            logger.debug(f"response: {response.json()}")
            logger.info(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                # Cache the response
                set_cached_response(cache_key, response.json())
                return response.json()
            else:
                return {
                    "error": f"API request failed: {response.status_code} - {response.text}",
                    "status": "error"
                }
                
        except requests.exceptions.RequestException as e:
            return {
                "error": f"Network error occurred: {str(e)}",
                "status": "error"
            }
        except Exception as e:
            return {
                "error": f"An error occurred: {str(e)}",
                "status": "error"
            }
    
    async def send_chat_message_stream(self, messages: List[Dict], session_id: Optional[str] = None, model: str = DEFAULT_MODEL, msg: cl.Message = None):
        """Send chat message and stream the response"""
        try:
            # Get the full response first
            response = await self.send_chat_message(messages, session_id, model)
            
            if msg and response.get("status") != "error":
                message_content = response.get("message", "")
                
                if message_content:
                    # Stream the response word by word for better UX
                    words = message_content.split(' ')
                    for i, word in enumerate(words):
                        if i == 0:
                            await msg.stream_token(word)
                        else:
                            await msg.stream_token(' ' + word)
                        # Small delay to simulate streaming
                        import asyncio
                        await asyncio.sleep(0.05)
                else:
                    await msg.stream_token("No message content received.")
            elif msg and response.get("status") == "error":
                await msg.stream_token(f"❌ Error: {response.get('error', 'Unknown error occurred')}")
            
            return response
            
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            if msg:
                await msg.stream_token(error_msg)
            return {"error": error_msg, "status": "error"}
    
    async def get_available_models(self) -> Dict:
        """Fetch available models from the API"""
        try:
            headers = {
                'Content-Type': 'application/json'
            }
            
            response = requests.get(
                self.models_endpoint,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": f"Failed to fetch models: {response.status_code} - {response.text}",
                    "models": {DEFAULT_MODEL: DEFAULT_MODEL}  # Fallback
                }
                
        except requests.exceptions.RequestException as e:
            return {
                "error": f"Network error fetching models: {str(e)}",
                "models": {DEFAULT_MODEL: DEFAULT_MODEL}  # Fallback
            }
        except Exception as e:
            return {
                "error": f"Error fetching models: {str(e)}",
                "models": {DEFAULT_MODEL: DEFAULT_MODEL}  # Fallback
            }
        
# Initialize conversational chat client
chat_client = ConversationalChatClient(API_BASE_URL)

async def display_search_results_table(results_list: List[Dict], total_count: int):
    """Display search results in a formatted table with interactive elements"""
    try:
        # Create table header
        table_content = f"## 📊 Search Results ({total_count} found)\n\n"
        
        if not results_list:
            table_content += "No results to display."
            await cl.Message(content=table_content).send()
            return
        
        # Determine table columns based on available data
        sample_result = results_list[0]
        available_fields = list(sample_result.keys())
        
        # Common field mappings and priorities
        field_priority = {
            'company_name': 1,
            'name': 1,
            'title': 2,
            'company': 2,
            'industry': 3,
            'location': 4,
            'city': 4,
            'state': 4,
            'country': 4,
            'company_size': 5,
            'revenue': 5,
            'company_website': 6,
            'linkedin_url': 7,
            'email': 8,
            'phone': 9
        }
        
        # Select and sort fields for display
        display_fields = []
        for field in available_fields:
            if field.lower() in field_priority:
                display_fields.append((field, field_priority[field.lower()]))
            else:
                display_fields.append((field, 10))  # Default priority
        
        # Sort by priority and limit to top 6 fields for readability
        display_fields.sort(key=lambda x: x[1])
        display_fields = [field[0] for field in display_fields[:6]]
        
        # Create table header
        header_row = "| " + " | ".join([field.replace('_', ' ').title() for field in display_fields]) + " |"
        separator_row = "|" + "|".join([" --- " for _ in display_fields]) + "|"
        
        table_content += header_row + "\n" + separator_row + "\n"
        
        # Add table rows
        for i, result in enumerate(results_list[:20]):  # Limit to first 20 results for performance
            row_data = []
            for field in display_fields:
                value = result.get(field, "N/A")
                
                # Handle different data types and create links where appropriate
                if isinstance(value, (list, dict)):
                    value = str(value)
                elif value is None:
                    value = "N/A"
                else:
                    value = str(value)
                
                # Create clickable links for URLs
                if field in ['company_website', 'linkedin_url'] and value and value != "N/A":
                    if not value.startswith('http'):
                        if field == 'company_website':
                            value = f"https://{value}"
                        elif field == 'linkedin_url' and not value.startswith('linkedin.com'):
                            value = f"https://linkedin.com{value}" if value.startswith('/') else f"https://linkedin.com/in/{value}"
                    
                    # Create markdown link
                    display_text = value.replace('https://', '').replace('http://', '')[:30]
                    if len(display_text) < len(value.replace('https://', '').replace('http://', '')):
                        display_text += "..."
                    value = f"[{display_text}]({value})"
                
                # Handle email links
                elif field == 'email' and value and value != "N/A" and '@' in value:
                    value = f"[{value}](mailto:{value})"
                
                # Truncate long text
                if len(value) > 50 and not value.startswith('['):
                    value = value[:47] + "..."
                
                row_data.append(value)
            
            table_row = "| " + " | ".join(row_data) + " |"
            table_content += table_row + "\n"
        
        # Add footer if there are more results
        if total_count > len(results_list):
            table_content += f"\n*Showing {len(results_list)} of {total_count} results*"
        
        if len(results_list) > 20:
            table_content += f"\n*Table shows first 20 results of {len(results_list)} returned*"
        
        # Send the table as a message
        await cl.Message(
            content=table_content
        ).send()
        
        # Also create an action button for exporting results
        actions = [
            cl.Action(
                name="export_results",
                value="export_results",
                payload={"results": results_list},
                label="📥 Export Results",
                description="Export search results as JSON"
            )
        ]
        
        await cl.Message(
            content="**Actions:**",
            actions=actions
        ).send()
        
    except Exception as e:
        await cl.Message(
            content=f"❌ **Error displaying results:** {str(e)}"
        ).send()

@cl.action_callback("export_results")
async def on_export_results(action: cl.Action):
    """Handle export results action"""
    try:
        results_data = action.payload.get("results", [])
        
        # Create a formatted JSON string
        formatted_json = json.dumps(results_data, indent=2)
        
        # Send as a text element that can be downloaded
        text_element = cl.Text(
            name="search_results.json",
            content=formatted_json,
            display="side"
        )
        
        await cl.Message(
            content="📥 **Search Results Export:**",
            elements=[text_element]
        ).send()
        
    except Exception as e:
        await cl.Message(
            content=f"❌ **Error exporting results:** {str(e)}"
        ).send()

@cl.action_callback("select_model")
async def on_select_model(action: cl.Action):
    """Handle model selection"""
    try:
        # Debug: Print action attributes
        logger.debug(f"Action attributes: {dir(action)}")
        logger.info(f"Action payload: {action.payload}")
        
        # Get model info from payload
        model_info = action.payload
        model_key = model_info.get("model")
        model_name = model_info.get("name", model_key)
        
        if not model_key:
            raise ValueError("Model key not found in action payload")
        
        # Update selected model in session
        cl.user_session.set("selected_model", model_key)
        
        await cl.Message(
            content=f"✅ **Model Updated!**\n\n"
                   f"Now using: **{model_name}** (`{model_key}`)\n\n"
                   f"Your conversations will now use this model. Continue chatting! 🚀"
        ).send()
        
    except Exception as e:
        logger.error(f"Error in model selection: {str(e)}")
        await cl.Message(
            content=f"❌ **Error selecting model:** {str(e)}"
        ).send()

async def display_results_sidebar(results_list: List[Dict], total_count: int):
    """Display search results table in the sidebar"""
    try:
        logger.info(f"display_results_sidebar called: results={len(results_list)}, total_count={total_count}")
        
        # Store results in session for export
        cl.user_session.set("search_results", results_list)
        cl.user_session.set("results_total_count", total_count)
        
        # Force a small delay to ensure proper sidebar clearing
        import asyncio
        await asyncio.sleep(0.1)
        
        # Create the table
        table_content = f"## 📊 Search Results\n"
        table_content += f"**Total Results:** {total_count} | **Showing:** {len(results_list)} results\n\n"
        
        if not results_list:
            table_content += "No results to display."
        else:
            # Determine table columns based on available data
            sample_result = results_list[0]
            available_fields = list(sample_result.keys())
            
            # Common field mappings and priorities
            field_priority = {
                'company_name': 1,
                'name': 1,
                'title': 2,
                'company': 2,
                'industry': 3,
                'location': 4,
                'city': 4,
                'state': 4,
                'country': 4,
                'company_size': 5,
                'revenue': 5,
                'company_website': 6,
                'website': 6,
                'linkedin_url': 7,
                'email': 8,
                'phone': 9
            }
            
            # Select and sort fields for display
            display_fields = []
            for field in available_fields:
                if field.lower() in field_priority:
                    display_fields.append((field, field_priority[field.lower()]))
                else:
                    display_fields.append((field, 10))  # Default priority
            
            # Sort by priority and limit to top 7 fields for sidebar
            display_fields.sort(key=lambda x: x[1])
            display_fields = [field[0] for field in display_fields[:7]]
            
            # Create table header
            header_row = "| " + " | ".join([field.replace('_', ' ').title() for field in display_fields]) + " |"
            separator_row = "|" + "|".join([" --- " for _ in display_fields]) + "|"
            
            table_content += header_row + "\n" + separator_row + "\n"
            
            # Add table rows for all results
            for i, result in enumerate(results_list):
                row_data = []
                for field in display_fields:
                    value = result.get(field, "N/A")
                    
                    # Handle different data types and create links where appropriate
                    if isinstance(value, (list, dict)):
                        value = str(value)
                    elif value is None:
                        value = "N/A"
                    else:
                        value = str(value)
                    
                    # Create clickable links for URLs
                    if field in ['website', 'company_website', 'linkedin_url', 'linkedin'] and value and value != "N/A":
                        if not value.startswith('http'):
                            if field in ['company_website', 'website']:
                                value = f"https://{value}"
                            elif field in ['linkedin_url', 'linkedin'] and not value.startswith('linkedin.com'):
                                value = f"https://linkedin.com{value}" if value.startswith('/') else f"https://linkedin.com/in/{value}"
                        
                        # Create markdown link with shorter display text for sidebar
                        display_text = value.replace('https://', '').replace('http://', '')[:15]
                        if len(display_text) < len(value.replace('https://', '').replace('http://', '')):
                            display_text += "..."
                        value = f"[{display_text}]({value})"
                    
                    # Handle email links
                    elif field == 'email' and value and value != "N/A" and '@' in value:
                        value = f"[{value}](mailto:{value})"
                    
                    row_data.append(value)
                
                table_row = "| " + " | ".join(row_data) + " |"
                table_content += table_row + "\n"
        
        # Create sidebar elements with just the table - use unique name to force update
        import time
        unique_id = str(int(time.time()))
        
        logger.info(f"Updating sidebar with {len(results_list)} results, unique_id: {unique_id}")
        
        elements = [
            cl.Text(
                name=f"search_results_table_{unique_id}",
                content=table_content
            )
        ]
        
        # Clear sidebar first, then set new elements to force update
        try:
            await cl.ElementSidebar.set_elements([])
            logger.info("Sidebar cleared")
            await cl.ElementSidebar.set_elements(elements)
            logger.info(f"Sidebar updated with new table (id: {unique_id})")
        except Exception as e:
            logger.error(f"Error updating sidebar: {e}")
        
        # Send export buttons as a pinned message at the top
        export_msg = await cl.Message(
            content=f"📥 **Export {len(results_list)} results:**",
            actions=[
                cl.Action(
                    name="export_csv",
                    value="export_csv",
                    payload={"format": "csv"},
                    label="📥 CSV",
                    description="Download as CSV"
                ),
                cl.Action(
                    name="export_json",
                    value="export_json",
                    payload={"format": "json"},
                    label="📄 JSON", 
                    description="Download as JSON"
                )
            ]
        ).send()
        
        # Pin the export message so it stays visible
        if hasattr(export_msg, 'id'):
            cl.user_session.set("export_message_id", export_msg.id)
        
    except Exception as e:
        logger.error(f"Error displaying sidebar table: {str(e)}")
        await cl.Message(
            content=f"❌ **Error displaying sidebar table:** {str(e)}"
        ).send()

@cl.action_callback("export_csv")
async def on_export_csv(action: cl.Action):
    """Handle CSV export"""
    try:
        import csv
        import io
        from datetime import datetime
        
        # Get stored results
        results_list = cl.user_session.get("search_results", [])
        total_count = cl.user_session.get("results_total_count", 0)
        
        logger.info(f"CSV Export: Found {len(results_list)} results in session, total_count={total_count}")
        
        if not results_list:
            await cl.Message(content="❌ **Error:** No search results to export.").send()
            return
        
        # Create CSV content
        output = io.StringIO()
        
        # Get all possible field names
        all_fields = set()
        for result in results_list:
            all_fields.update(result.keys())
        
        logger.info(f"CSV Export: Found {len(all_fields)} unique fields")
        
        # Sort fields for consistent column order
        field_priority = {
            'company_name': 1, 'name': 1, 'title': 2, 'company': 2, 'industry': 3,
            'location': 4, 'city': 4, 'state': 4, 'country': 4, 'company_size': 5,
            'revenue': 5, 'website': 6, 'company_website': 6, 'linkedin_url': 7,
            'email': 8, 'phone': 9
        }
        
        sorted_fields = sorted(all_fields, key=lambda x: field_priority.get(x.lower(), 10))
        
        # Write CSV
        writer = csv.DictWriter(output, fieldnames=sorted_fields, extrasaction='ignore')
        writer.writeheader()
        
        for result in results_list:
            # Clean up data for CSV
            clean_result = {}
            for field in sorted_fields:
                value = result.get(field, "")
                if isinstance(value, (list, dict)):
                    value = str(value)
                elif value is None:
                    value = ""
                clean_result[field] = value
            writer.writerow(clean_result)
        
        csv_content = output.getvalue()
        output.close()
        
        logger.info(f"CSV Export: Generated CSV with {len(csv_content)} characters")
        
        # Create downloadable file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"lead_search_results_{timestamp}.csv"
        
        # Send as downloadable text element
        text_element = cl.File(
            name=filename,
            content=csv_content.encode('utf-8'),
            display="inline"
        )
        
        await cl.Message(
            content=f"📥 **CSV Export Complete!**\n\n"
                   f"**File:** {filename}\n"
                   f"**Records:** {len(results_list)}\n"
                   f"**Columns:** {len(sorted_fields)}\n"
                   f"**Total Available:** {total_count} records",
            elements=[text_element]
        ).send()
        
    except Exception as e:
        logger.error(f"Error in CSV export: {str(e)}")
        await cl.Message(content=f"❌ **Error exporting CSV:** {str(e)}").send()

@cl.action_callback("export_json")
async def on_export_json(action: cl.Action):
    """Handle JSON export"""
    try:
        from datetime import datetime
        
        # Get stored results
        results_list = cl.user_session.get("search_results", [])
        total_count = cl.user_session.get("results_total_count", 0)
        
        if not results_list:
            await cl.Message(content="❌ **Error:** No search results to export.").send()
            return
        
        # Create JSON structure with metadata
        export_data = {
            "export_info": {
                "timestamp": datetime.now().isoformat(),
                "total_results": total_count,
                "exported_results": len(results_list)
            },
            "results": results_list
        }
        
        # Create formatted JSON string
        formatted_json = json.dumps(export_data, indent=2, ensure_ascii=False)
        
        # Create downloadable file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"lead_search_results_{timestamp}.json"
        
        # Send as downloadable file element
        text_element = cl.File(
            name=filename,
            content=formatted_json.encode('utf-8'),
            display="inline"
        )
        
        await cl.Message(
            content=f"📄 **JSON Export Complete!**\n\n"
                   f"**File:** {filename}\n"
                   f"**Records:** {len(results_list)}\n"
                   f"**Size:** {len(formatted_json):,} characters",
            elements=[text_element]
        ).send()
        
    except Exception as e:
        await cl.Message(content=f"❌ **Error exporting JSON:** {str(e)}").send()

@cl.on_chat_start
async def start_chat():
    """Initialize the chat session"""
    # Generate a unique session ID for this chat
    session_id = str(uuid.uuid4())
    
    # Initialize session state
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("message_history", [])
    cl.user_session.set("selected_model", DEFAULT_MODEL)
    
    # Check API configuration
    if not API_BASE_URL or API_BASE_URL == "https://your-api-base-url.com":
        await cl.Message(
            content="⚠️ **Configuration Warning:**\n\n"
                   "API_BASE_URL environment variable is not set properly.\n"
                   "Please add your API base URL to your .env file:\n\n"
                   "```\nAPI_BASE_URL=https://your-api-domain.com\n```"
        ).send()
        return
    
    # Fetch available models
    models_response = await chat_client.get_available_models()
    available_models = models_response.get("models", {DEFAULT_MODEL: DEFAULT_MODEL})
    
    if models_response.get("error"):
        await cl.Message(
            content=f"⚠️ **Warning:** {models_response.get('error')}\n\nUsing default model: {DEFAULT_MODEL}"
        ).send()
    
    # Store available models in session
    cl.user_session.set("available_models", available_models)
    
    # Create model selection actions
    model_actions = []
    for model_key, model_name in available_models.items():
        is_default = model_key == DEFAULT_MODEL
        label = f"🤖 {model_name}" + (" (Default)" if is_default else "")
        model_actions.append(
            cl.Action(
                name="select_model",
                value=model_key,
                payload={"model": model_key, "name": model_name},
                label=label,
                description=f"Use {model_name} for conversations"
            )
        )
    
    # Welcome message with model selection
    welcome_content = (f"🚀 **Conversational Lead Generation Assistant**\n\n"
                      f"I'm here to help you through a natural conversation to understand your lead generation needs.\n\n"
                      f"**Session ID:** `{session_id}`\n"
                      f"**Current Model:** `{DEFAULT_MODEL}`\n\n"
                      f"**What I can help with:**\n"
                      f"• Extract and understand your lead criteria\n"
                      f"• Search for relevant leads based on your requirements\n"
                      f"• Provide insights and recommendations\n"
                      f"• Maintain conversation context throughout our session\n\n"
                      f"**Choose your AI model below or start chatting!** 💡")
    
    await cl.Message(
        content=welcome_content,
        actions=model_actions
    ).send()
    
    # Instructions
    await cl.Message(
        content="💬 **Ready to start!** Tell me about the type of leads you're looking for, or click a model above to change it first."
    ).send()

async def process_conversation_message(user_input: str):
    """Process user message through the conversational API"""
    session_id = cl.user_session.get("session_id")
    message_history = cl.user_session.get("message_history", [])
    selected_model = cl.user_session.get("selected_model", DEFAULT_MODEL)
    
    # Prepare messages including conversation history
    messages = message_history.copy()
    messages.append({"role": "user", "content": user_input})
    
    # Create a streaming message
    msg = cl.Message(content="")
    await msg.send()
    
    try:
        # Send message to conversational API with streaming using selected model
        response = await chat_client.send_chat_message_stream(
            messages=messages, 
            session_id=session_id, 
            model=selected_model,
            msg=msg
        )
        
        # Handle the response
        if response.get("status") != "error":
            # Update conversation history
            assistant_message = response.get("message", "")
            message_history.append({"role": "user", "content": user_input})
            message_history.append({"role": "assistant", "content": assistant_message})
            cl.user_session.set("message_history", message_history)
            
            # Show additional information if available
            additional_info = []
            
            if response.get("ready_for_search"):
                additional_info.append("✅ **Ready for search** - I have enough information to find leads")
            
            if response.get("missing_info"):
                missing_info = response.get("missing_info")
                if isinstance(missing_info, list):
                    missing_info = ", ".join(missing_info)
                additional_info.append(f"📝 **Missing info:** {missing_info}")
            
            if response.get("search_results") and response.get("search_results", {}).get("search_performed"):
                search_results = response.get("search_results")
                results_count = search_results.get("count", 0)
                results_list = search_results.get("results", [])
                additional_info.append(f"🔍 **Search results:** Found {results_count} results")
                
                # Display search results in sidebar only
                if results_list:
                    # Clear previous search results from session to ensure fresh state
                    cl.user_session.set("search_results", [])
                    cl.user_session.set("results_total_count", 0)
                    
                    print(f"New search results: {len(results_list)} results, clearing sidebar for fresh update")
                    await display_results_sidebar(results_list, results_count)

            if response.get("session_summary"):
                additional_info.append(f"📋 **Session summary:** {response.get('session_summary')}")
            
            if response.get("domain_check"):
                domain_info = response.get("domain_check")
                if isinstance(domain_info, dict):
                    domain_status = domain_info.get("status", "unknown")
                    additional_info.append(f"🌐 **Domain check:** {domain_status}")
                else:
                    additional_info.append(f"🌐 **Domain check:** {domain_info}")
            
            if response.get("intent_analysis"):
                intent_info = response.get("intent_analysis")
                if isinstance(intent_info, dict):
                    intent_type = intent_info.get("intent", "unknown")
                    confidence = intent_info.get("confidence", "")
                    if confidence:
                        additional_info.append(f"🎯 **Intent analysis:** {intent_type} (confidence: {confidence})")
                    else:
                        additional_info.append(f"🎯 **Intent analysis:** {intent_type}")
                else:
                    additional_info.append(f"🎯 **Intent analysis:** {intent_info}")
            
            if response.get("extracted_criteria"):
                criteria_info = response.get("extracted_criteria")
                if isinstance(criteria_info, dict):
                    criteria_count = len([k for k, v in criteria_info.items() if v])
                    additional_info.append(f"📝 **Extracted criteria:** {criteria_count} criteria identified")
                else:
                    additional_info.append(f"📝 **Extracted criteria:** {criteria_info}")
            
            if response.get("tool_metadata"):
                tool_info = response.get("tool_metadata")
                if tool_info != "no search yet":
                    if isinstance(tool_info, dict):
                        filter_count = len([k for k, v in tool_info.items() if v])
                        additional_info.append(f"🔧 **Search filters:** {tool_info.get("filters", "N/A")} filters applied")
                        additional_info.append(f"🔧 **Service used:** {tool_info.get("service", "N/A")}")
                    else:
                        additional_info.append(f"🔧 **Tool metadata:** {tool_info}")
            
            # Send additional information if available
            if additional_info:
                await cl.Message(
                    content="\n\n---\n\n" + "\n\n".join(additional_info)
                ).send()
        
        # Update the streaming message to finalize
        await msg.update()
        
    except Exception as e:
        await msg.update(content=f"❌ **Error:** Unable to process your request: {str(e)}")

@cl.on_message
async def main(message: cl.Message):
    """Main message handler"""
    user_input = message.content
    
    # Check for model change command
    if user_input.lower().startswith('/models') or user_input.lower().startswith('/change-model'):
        await show_model_selection()
        return
    
    # Check for current model info command
    if user_input.lower().startswith('/model-info'):
        await show_current_model_info()
        return
    
    # Check for cache info command
    if user_input.lower().startswith('/cache-info'):
        await show_cache_info()
        return
    
    # Clear cache command
    if user_input.lower().startswith('/clear-cache'):
        await clear_cache()
        return
    
    # Process the message through the conversational API
    await process_conversation_message(user_input)

async def show_model_selection():
    """Show available models for selection"""
    try:
        available_models = cl.user_session.get("available_models", {DEFAULT_MODEL: DEFAULT_MODEL})
        current_model = cl.user_session.get("selected_model", DEFAULT_MODEL)
        
        # Create model selection actions
        model_actions = []
        for model_key, model_name in available_models.items():
            is_current = model_key == current_model
            label = f"🤖 {model_name}" + (" (Current)" if is_current else "")
            model_actions.append(
                cl.Action(
                    name="select_model",
                    value=model_key,
                    payload={"model": model_key, "name": model_name},
                    label=label,
                    description=f"Switch to {model_name}"
                )
            )
        
        await cl.Message(
            content=f"🤖 **Available Models**\n\nCurrent: **{available_models.get(current_model, current_model)}**\n\nSelect a different model:",
            actions=model_actions
        ).send()
        
    except Exception as e:
        await cl.Message(
            content=f"❌ **Error showing models:** {str(e)}"
        ).send()

async def show_current_model_info():
    """Show current model information"""
    try:
        available_models = cl.user_session.get("available_models", {DEFAULT_MODEL: DEFAULT_MODEL})
        current_model = cl.user_session.get("selected_model", DEFAULT_MODEL)
        current_model_name = available_models.get(current_model, current_model)
        
        await cl.Message(
            content=f"🤖 **Current Model Information**\n\n"
                   f"**Model ID:** `{current_model}`\n"
                   f"**Model Name:** {current_model_name}\n\n"
                   f"**Commands:**\n"
                   f"• `/models` or `/change-model` - Change model\n"
                   f"• `/model-info` - Show this information"
        ).send()
        
    except Exception as e:
        await cl.Message(
            content=f"❌ **Error showing model info:** {str(e)}"
        ).send()

async def show_cache_info():
    """Show cache information"""
    try:
        global api_cache
        cache_count = len(api_cache)
        
        if cache_count == 0:
            cache_size = 0
            file_exists = CACHE_FILE.exists()
            await cl.Message(
                content=f"💾 **Cache Information**\n\n"
                       f"**Status:** Empty\n"
                       f"**Entries:** 0\n"
                       f"**File:** {'Exists' if file_exists else 'Not created'}\n"
                       f"**Location:** `{CACHE_FILE}`\n"
                       f"**Duration:** {CACHE_DURATION//60} minutes per entry"
            ).send()
        else:
            # Calculate cache stats
            current_time = time.time()
            active_entries = 0
            expired_entries = 0
            
            for cache_key, cache_data in api_cache.items():
                if current_time - cache_data["timestamp"] < CACHE_DURATION:
                    active_entries += 1
                else:
                    expired_entries += 1
            
            # Get file size
            try:
                cache_size = CACHE_FILE.stat().st_size if CACHE_FILE.exists() else 0
                cache_size_mb = cache_size / (1024 * 1024)
            except:
                cache_size_mb = 0
            
            await cl.Message(
                content=f"💾 **Cache Information**\n\n"
                       f"**Total Entries:** {cache_count}\n"
                       f"**Active:** {active_entries}\n"
                       f"**Expired:** {expired_entries}\n"
                       f"**File Size:** {cache_size_mb:.2f} MB\n"
                       f"**Location:** `{CACHE_FILE}`\n"
                       f"**Duration:** {CACHE_DURATION} seconds ({CACHE_DURATION//60} minutes)\n\n"
                       f"**Commands:**\n"
                       f"• `/cache-info` - Show this information\n"
                       f"• `/clear-cache` - Clear all cached responses\n"
                       f"• `/cleanup-cache` - Remove only expired entries"
            ).send()
            
    except Exception as e:
        await cl.Message(
            content=f"❌ **Error showing cache info:** {str(e)}"
        ).send()

async def clear_cache():
    """Clear all cached responses"""
    try:
        global api_cache
        cache_count = len(api_cache)
        api_cache.clear()
        
        # Remove cache file
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()
        
        await cl.Message(
            content=f"🗑️ **Cache Cleared!**\n\n"
                   f"Removed {cache_count} cached entries.\n"
                   f"Deleted cache file: `{CACHE_FILE}`\n"
                   f"Next API requests will be fresh calls."
        ).send()
        
    except Exception as e:
        await cl.Message(
            content=f"❌ **Error clearing cache:** {str(e)}"
        ).send()

# Run the app
if __name__ == "__main__":
    cl.run()