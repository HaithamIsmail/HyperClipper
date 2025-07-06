import gradio as gr
import lancedb
import os
from video_pipeline import run_pipeline
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import base64
import tempfile
import shutil

from utils import get_text_embedding, sample_from_video, convert_image_to_base64
from config import load_config
from lancedb_utils import retreive_clip
from gradio import ChatMessage

app_config = load_config()
langchain_message_history = []

chat_model = ChatOpenAI(
    # model="Qwen/Qwen3-30B-A3B",
    model="Qwen/Qwen3-32B",
    base_url="https://api.studio.nebius.com/v1/",
    api_key=app_config.NEBIUS_API_KEY.get_secret_value()
)

chat_model_vlm = ChatOpenAI(
    model="Qwen/Qwen2.5-VL-32B-Instruct",
    base_url=app_config.MODAL_VLM_URL.get_secret_value(),
    api_key=app_config.MODEL_API_KEY.get_secret_value()
)

def search_clips(query_text, limit=3):
    """Searches the LanceDB database for clips matching the query."""
    try:
        # Create embedding for the query using Hugging Face API
        query_vector = get_text_embedding(query_text, app_config.CLIP_EMBEDDING_URL.get_secret_value())[0]
        
        # Connect to LanceDB
        db = lancedb.connect(app_config.LANCEDB_URI.get_secret_value())
        table = db.open_table("video_clips")
        
        # Search for similar clips
        results = table.search(query_vector).limit(limit).to_pandas()
        return results
        
    except FileNotFoundError:
        return f"Error: Database not found at {app_config.LANCEDB_URI.get_secret_value()}. Please ensure the video analysis server has processed some videos first."
    except Exception as e:
        return f"Error during search: {str(e)}"

def format_search_results(results_df):
    """Format search results for display."""
    if isinstance(results_df, str):  # Error message
        return results_df
    
    if results_df.empty:
        return "No clips found matching your query."
    
    response = "Here are the top results I found:\n\n"
    for idx, row in results_df.iterrows():
        response += f"**Clip {row.get('clip_id', 'N/A')} from {row.get('video_name', 'Unknown')}**\n"
        response += f"⏰ Time: {row.get('start_time', 'N/A')}s - {row.get('end_time', 'N/A')}s\n"
        
        # Handle summary safely
        summary = row.get('summary', 'No summary available')
        if isinstance(summary, str) and '---' in summary:
            summary = summary.split('---')[0].strip()
        
        response += f"📝 Summary: {summary}\n"
        
        # Add score if available
        if '_distance' in row:
            score = 1 - row['_distance']  # Convert distance to similarity score
            response += f"🎯 Relevance: {score:.2f}\n"
        
        response += "\n---\n\n"
    
    return response

def get_clip_videos_and_thumbnails(results_df):
    """Extract video clips and thumbnails from search results."""
    if isinstance(results_df, str) or results_df.empty:
        return [], []
    
    videos = []
    thumbnails = []
    
    for idx, row in results_df.iterrows():
        # Get video clip path
        clip_path = row.get('clip_path', '')
        if clip_path and os.path.exists(clip_path):
            videos.append(clip_path)
        else:
            videos.append(None)
        
        # Get thumbnail from base64
        thumbnail_b64 = row.get('thumbnail', '')
        if thumbnail_b64:
            try:
                # Decode base64 thumbnail and save as temp file
                thumbnail_data = base64.b64decode(thumbnail_b64)
                temp_thumb = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                temp_thumb.write(thumbnail_data)
                temp_thumb.close()
                thumbnails.append(temp_thumb.name)
            except Exception as e:
                print(f"Error processing thumbnail: {e}")
                thumbnails.append(None)
        else:
            thumbnails.append(None)
    
    return videos, thumbnails

# Global state to store the latest search results for the UI
latest_search_results = {"results": None, "query": "", "clips_display": []}

@tool
def get_relevant_clips(query):
    """Retrieve relevant clips from vector database

    Args:
        query: Text to use in vector search

    Returns :
        str: the search results formatted in a string
    """
    global latest_search_results
    
    search_result = search_clips(query, limit=5)
    formatted_search_result = format_search_results(search_result)
    
    # Store the results globally so the UI can access them
    latest_search_results["results"] = search_result
    latest_search_results["query"] = query
    
    # Prepare clips display data
    if not isinstance(search_result, str) and not search_result.empty:
        videos, thumbnails = get_clip_videos_and_thumbnails(search_result)
        clip_components = []
        
        for idx, (row_idx, row) in enumerate(search_result.iterrows()):
            video = videos[idx] if idx < len(videos) else None
            thumbnail = thumbnails[idx] if idx < len(thumbnails) else None
            
            if video or thumbnail:  # Only show if we have media
                info = {
                    'clip_id': row.get('clip_id', 'N/A'),
                    'video_name': row.get('video_name', 'Unknown'),
                    'start_time': row.get('start_time', 'N/A'),
                    'end_time': row.get('end_time', 'N/A'),
                    'summary': row.get('summary', '').split('---')[0].strip() if '---' in str(row.get('summary', '')) else row.get('summary', ''),
                    'relevance': 1 - row['_distance'] if '_distance' in row else 0
                }
                clip_components.append({
                    'video': video,
                    'thumbnail': thumbnail,
                    'info': info
                })
        
        latest_search_results["clips_display"] = clip_components
    else:
        latest_search_results["clips_display"] = []

    return formatted_search_result

@tool
def get_clip(clip_id: str):
    """Retreive the clip

    Args:
        clip_id: id of the clip to retreive

    Returns :
        list: list of frames
    """
    print("clip id", clip_id)
    clip = retreive_clip(clip_id, app_config.LANCEDB_URI.get_secret_value())
    images = sample_from_video(clip["clip_path"])
    base64_images = [convert_image_to_base64(image, "png") for image in images]
    return base64_images

def search_and_display_clips(query_text):
    """Search for clips and return both formatted text and video/thumbnail data."""
    search_results = search_clips(query_text, limit=5)
    formatted_results = format_search_results(search_results)
    
    if isinstance(search_results, str):  # Error case
        return formatted_results, [], []
    
    videos, thumbnails = get_clip_videos_and_thumbnails(search_results)
    
    # Prepare clip info for display
    clip_info = []
    for idx, row in search_results.iterrows():
        info = {
            'clip_id': row.get('clip_id', 'N/A'),
            'video_name': row.get('video_name', 'Unknown'),
            'start_time': row.get('start_time', 'N/A'),
            'end_time': row.get('end_time', 'N/A'),
            'summary': row.get('summary', '').split('---')[0].strip() if '---' in str(row.get('summary', '')) else row.get('summary', ''),
            'relevance': 1 - row['_distance'] if '_distance' in row else 0
        }
        clip_info.append(info)
    
    return formatted_results, videos, thumbnails, clip_info

def chat_agent(message, history: list):
    """Core agent logic function."""
    global latest_search_results, langchain_message_history

    # Add current message
    langchain_message_history.append({"role": "user", "content": message})
    
    llm_with_tool = chat_model.bind_tools(tools=[get_relevant_clips])
    tools = {"get_relevant_clips": get_relevant_clips}

    # The agent loop
    while True:
        ai_response = llm_with_tool.invoke(langchain_message_history)

        if not ai_response.tool_calls:
            break

        for tool_call in ai_response.tool_calls:
            tool_output = tools[tool_call["name"]].invoke(tool_call)
            tool_call_log = {
                "role": "tool",
                "tool_call_id": tool_output.tool_call_id,
                "content": tool_output.content
            }
            langchain_message_history.append(tool_call_log)
    
    content = ai_response.content
    if "</think>" in content:
        content = content.split("</think>")[-1].strip()
    
    # The global state `latest_search_results` is updated by the tool.
    # The text response is returned.
    langchain_message_history.append({"role": "assistant", "content": content})
    return langchain_message_history

def chat_agent_mm(message, history):
    """Core agent logic function."""
    global latest_search_results, langchain_message_history
    
    langchain_message_history.append({"role": "user", "content": message})
    history.append({"role": "user", "content": message})
    
    print(langchain_message_history)
    llm_with_tool = chat_model_vlm.bind_tools(tools=[get_relevant_clips, get_clip])
    tools = {"get_relevant_clips": get_relevant_clips, "get_clip": get_clip}

    # The agent loop
    while True:
        ai_response = llm_with_tool.invoke(langchain_message_history)

        if not ai_response.tool_calls:
            break

        for tool_call in ai_response.tool_calls:
            print(tool_call)
            langchain_message_history.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        tool_call                
                    ]
                }
            )
            history.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        tool_call                
                    ]
                }
            )
            tool_output = tools[tool_call["name"]].invoke(tool_call)
            if tool_call["name"] == "get_clip":
                tool_call_log = {
                    "role": "tool",
                    "tool_call_id": tool_output.tool_call_id,
                    "content": "retrieved clip will be provided by the user after this message"
                }
                history.append(tool_call_log)
                langchain_message_history.extend([
                tool_call_log,
                {
                    "role": "user", "content": [
                        {"type": "text", "text": "here is the clip retreived by the tool"},
                        *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/png;base64,{x}'}}, tool_output.content)
                    ],
                }])
            else:
                tool_call_log = {
                    "role": "tool",
                    "tool_call_id": tool_output.tool_call_id,
                    "content": tool_output.content
                }
                langchain_message_history.append(tool_call_log)
                history.append(tool_call_log)
    
    content = ai_response.content
    if "</think>" in content:
        content = content.split("</think>")[-1].strip()
    
    # The global state `latest_search_results` is updated by the tool.
    # The text response is returned.
    langchain_message_history.append({"role": "assistant", "content": content})
    history.append({"role": "assistant", "content": content})
    return history

def get_latest_clips_for_display():
    """Get the latest search results for display in the UI."""
    global latest_search_results
    return latest_search_results.get("clips_display", [])

def check_database_status():
    """Check if the database exists and has data."""
    try:
        db = lancedb.connect(app_config.LANCEDB_URI.get_secret_value())
        table_names = db.table_names()
        if "video_clips" not in table_names:
            return f"✅ Database connected, but 'video_clips' table not found. Analyze a video to create it."
        table = db.open_table("video_clips")
        count = len(table)
        return f"✅ Database connected. Found {count} video clips."
    except Exception as e:
        return f"❌ Database issue: {str(e)}"

def check_server_status():
    """Check if the MCP server is running."""
    # This check is illustrative; adjust if your server runs on a different port.
    return "➡️ To analyze videos, upload them in the 'Video Analyzer' tab."


# Create the Gradio interface
with gr.Blocks(title="Video Search Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Video Search Agent")
    gr.Markdown("Search through your processed video clips using natural language queries.")
    
    # Status section
    with gr.Accordion("System Status", open=False):
        status_text = gr.Textbox(
            label="Status",
            value=f"{check_database_status()}\n{check_server_status()}",
            interactive=False,
            lines=3
        )
        refresh_btn = gr.Button("Refresh Status")
        refresh_btn.click(
            fn=lambda: f"{check_database_status()}\n{check_server_status()}",
            outputs=status_text
        )

    # Chat interface with clips display
    with gr.Tab("💬 Chat with Clips"):
        # Manual chat layout for full control
        chatbot = gr.Chatbot(
            [],
            type="messages",
            label="Video Search Assistant",
            height=500,
            avatar_images=(None, "https://seeklogo.com/images/O/openai-logo-8284262873-seeklogo.com.png")
        )
        with gr.Row():
            chat_input = gr.Textbox(
                show_label=False,
                placeholder="Ask me to find clips about cooking...",
                lines=1,
                scale=4,
                container=False,
            )
            submit_btn = gr.Button("🔍 Search", variant="primary", scale=1, min_width=150)
        
        gr.Examples(
            [
                "find clips about cooking",
                "search for meeting discussions",
                "show me sports highlights", 
                "find outdoor activities"
            ],
            inputs=chat_input,
            label="Quick-search examples"
        )
            
        gr.Markdown("### 🎬 Found Clips")
        
        # State to store clips data for rendering
        clips_data_state = gr.State([])
        
        def handle_chat_and_clips(user_message, history):
            """Event handler for chat submission to update chat and clips."""
            new_history = chat_agent(user_message, history)
            # print(new_history)
            clips_data = get_latest_clips_for_display()
            return "", new_history, clips_data

        # Dynamic clip display
        @gr.render(inputs=[clips_data_state])
        def show_clips_in_chat(clip_data):
            # FIX: Wrap everything in a single gr.Column to ensure vertical stacking
            with gr.Column():
                if not clip_data:
                    gr.Markdown("*No clips found yet. Ask the assistant to search for something!*")
                    return
                
                gr.Markdown(f"**Found {len(clip_data)} relevant clips:**")

                for i, clip in enumerate(clip_data):
                    # Use a column for each clip block to keep them separate
                    with gr.Column(variant='panel'):
                        with gr.Row():
                            with gr.Column(scale=3):
                                # Clip info
                                info = clip['info']
                                gr.Markdown(f"**Clip {info['clip_id']}** from *{info['video_name']}*")
                                gr.Markdown(f"⏱️ {info['start_time']:.1f}s - {info['end_time']:.1f}s | 🎯 Relevance: {info['relevance']:.2f}")
                                
                                # Summary (shortened for chat view)
                                summary_text = info['summary'][:150] + "..." if len(info['summary']) > 150 else info['summary']
                                gr.Markdown(f"📝 {summary_text}")
                            
                            with gr.Column(scale=1):
                                # Video player (smaller for chat view)
                                if clip['video'] and os.path.exists(clip['video']):
                                    gr.Video(clip['video'], label="", height=180, show_label=False)
                                else:
                                    gr.Markdown("⚠️ *Video not available*")
        
        # Wire up submission events
        submit_btn.click(
            fn=handle_chat_and_clips,
            inputs=[chat_input, chatbot],
            outputs=[chat_input, chatbot, clips_data_state]
        )
        chat_input.submit(
            fn=handle_chat_and_clips,
            inputs=[chat_input, chatbot],
            outputs=[chat_input, chatbot, clips_data_state]
        )
        
    with gr.Tab("💬 Multimodal Chat with Clips"):
        # Manual chat layout for full control
        chatbot = gr.Chatbot(
            [],
            type="messages",
            label="Video Search Assistant",
            height=500,
            avatar_images=(None, "https://seeklogo.com/images/O/openai-logo-8284262873-seeklogo.com.png")
        )
        with gr.Row():
            chat_input = gr.Textbox(
                show_label=False,
                placeholder="Ask me to find clips about cooking...",
                lines=1,
                scale=4,
                container=False,
            )
            submit_btn = gr.Button("🔍 Search", variant="primary", scale=1, min_width=150)
        
        gr.Examples(
            [
                "search for clips about the number of computations in llms",
                "search for meeting discussions",
                "show me sports highlights", 
                "find outdoor activities"
            ],
            inputs=chat_input,
            label="Quick-search examples"
        )
            
        gr.Markdown("### 🎬 Found Clips")
        
        # State to store clips data for rendering
        clips_data_state = gr.State([])
        
        def handle_chat_and_clips(user_message, history):
            """Event handler for chat submission to update chat and clips."""
            new_history = chat_agent_mm(user_message, history)
            # print(new_history)
            clips_data = get_latest_clips_for_display()
            return "", new_history, clips_data

        # Dynamic clip display
        @gr.render(inputs=[clips_data_state])
        def show_clips_in_chat(clip_data):
            with gr.Column():
                if not clip_data:
                    gr.Markdown("*No clips found yet. Ask the assistant to search for something!*")
                    return
                
                gr.Markdown(f"**Found {len(clip_data)} relevant clips:**")

                for i, clip in enumerate(clip_data):
                    # Use a column for each clip block to keep them separate
                    with gr.Column(variant='panel'):
                        with gr.Row():
                            with gr.Column(scale=3):
                                # Clip info
                                info = clip['info']
                                gr.Markdown(f"**Clip {info['clip_id']}** from *{info['video_name']}*")
                                gr.Markdown(f"⏱️ {info['start_time']:.1f}s - {info['end_time']:.1f}s | 🎯 Relevance: {info['relevance']:.2f}")
                                
                                # Summary (shortened for chat view)
                                summary_text = info['summary'][:150] + "..." if len(info['summary']) > 150 else info['summary']
                                gr.Markdown(f"📝 {summary_text}")
                            
                            with gr.Column(scale=1):
                                # Video player (smaller for chat view)
                                if clip['video'] and os.path.exists(clip['video']):
                                    gr.Video(clip['video'], label="", height=180, show_label=False)
                                else:
                                    gr.Markdown("⚠️ *Video not available*")
        
        # Wire up submission events
        submit_btn.click(
            fn=handle_chat_and_clips,
            inputs=[chat_input, chatbot],
            outputs=[chat_input, chatbot, clips_data_state]
        )
        chat_input.submit(
            fn=handle_chat_and_clips,
            inputs=[chat_input, chatbot],
            outputs=[chat_input, chatbot, clips_data_state]
        )

    # Standalone search interface (keep the original for manual searching)
    with gr.Tab("🔍 Manual Search"):
        with gr.Row():
            with gr.Column(scale=2):
                search_input = gr.Textbox(
                    label="Search Query",
                    placeholder="Enter your search query (e.g., 'cooking scenes', 'meeting discussions')",
                    lines=2
                )
                search_btn = gr.Button("Search Clips", variant="primary")
                
                # Quick search examples
                gr.Markdown("**Quick Examples:**")
                example_buttons = []
                examples = [
                    "cooking scenes",
                    "meeting discussions", 
                    "sports highlights",
                    "outdoor activities"
                ]
                
                with gr.Row():
                    for example in examples:
                        btn = gr.Button(example, size="sm")
                        example_buttons.append(btn)
                        btn.click(fn=lambda x=example: x, outputs=search_input)
            
            with gr.Column(scale=1):
                search_results_text = gr.Textbox(
                    label="Search Results Summary",
                    lines=10,
                    max_lines=15,
                    interactive=False
                )
        
        # Clips display section
        gr.Markdown("## 🎬 Found Clips")
        clips_display = gr.Column(visible=False)
        
        # Store for clip data
        clips_state = gr.State([])
        
        def update_clips_display_manual(query):
            if not query.strip():
                return "Please enter a search query.", gr.Column(visible=False), []
            
            formatted_results, videos, thumbnails, clip_info = search_and_display_clips(query)
            
            if not videos or all(v is None for v in videos):
                return formatted_results, gr.Column(visible=False), []
            
            # Create clip display components
            clip_components = []
            for i, (video, thumbnail, info) in enumerate(zip(videos, thumbnails, clip_info)):
                if video or thumbnail:  # Only show if we have media
                    clip_components.append({
                        'video': video,
                        'thumbnail': thumbnail,
                        'info': info
                    })
            
            return formatted_results, gr.Column(visible=len(clip_components) > 0), clip_components
        
        # Update the display when search is triggered
        search_btn.click(
            fn=update_clips_display_manual,
            inputs=[search_input],
            outputs=[search_results_text, clips_display, clips_state]
        )
        
        # Dynamic clip display
        @gr.render(inputs=[clips_state])
        def show_clips(clip_data):
            if not clip_data:
                return
            
            for i, clip in enumerate(clip_data):
                with gr.Row():
                    with gr.Column():
                        # Clip info
                        info = clip['info']
                        gr.Markdown(f"### Clip {info['clip_id']} from {info['video_name']}")
                        gr.Markdown(f"**Time:** {info['start_time']}s - {info['end_time']}s")
                        gr.Markdown(f"**Relevance Score:** {info['relevance']:.2f}")
                        
                        # Summary
                        summary_text = info['summary'][:300] + "..." if len(info['summary']) > 300 else info['summary']
                        gr.Markdown(f"**Summary:** {summary_text}")
                        
                        # Video player
                        if clip['video'] and os.path.exists(clip['video']):
                            gr.Video(clip['video'], label="Play Clip")
                        else:
                            gr.Markdown("⚠️ Video file not available")
                
                gr.Markdown("---")
    
    # Video analyzer tool section
    with gr.Tab("📹 Video Analyzer"):
        gr.Markdown("""
        **To analyze new videos:**
        1. Upload your video file using the interface below
        2. Click "Analyze Video" to process the video
        3. The processed clips will be automatically added to the searchable database
        """)
        
        with gr.Row():
            video_file = gr.File(
                label="Upload Video",
                file_types=[".mp4"],
                type="filepath"
            )
            analyze_btn = gr.Button("Analyze Video", variant="primary")
        
        analysis_output = gr.Textbox(
            label="Analysis Status",
            lines=5,
            interactive=False
        )
        
        def analyze_video_local(file_obj):
            if not file_obj:
                return "Please select a video file first."
            try:
                # Save uploaded file to a temp file with the same name as the uploaded file
                if hasattr(file_obj, 'name'):
                    original_filename = os.path.basename(file_obj.name)
                else:
                    original_filename = "uploaded_video.mp4"
                temp_dir = tempfile.mkdtemp()
                tmp_path = os.path.join(temp_dir, original_filename)
                with open(tmp_path, "wb") as f:
                    f.write(file_obj)
                
                # Run the video processing pipeline
                run_pipeline(tmp_path)
                
                # Clean up temp file after processing
                try:
                    os.remove(tmp_path)
                    shutil.rmtree(temp_dir)
                except Exception:
                    pass
                
                return f"✅ Video analysis complete for '{original_filename}'. You can now search for clips from this video."
            except Exception as e:
                return f"❌ Error during video analysis: {str(e)}"
        
        analyze_btn.click(
            fn=analyze_video_local,
            inputs=[video_file],
            outputs=[analysis_output]
        )

# Launch the application
if __name__ == "__main__":
    print("🚀 Starting Video Search Agent...")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )