---
title: Clip Search Agent
emoji: 🎬
colorFrom: blue
colorTo: pink
sdk: gradio
sdk_version: 5.33.0
app_file: app.py
pinned: false
license: apache-2.0
tags:
    - agent-demo-track
short_description: An agent that allows chatting with parts of video
---

### This was a submission to Gradio Agents-MCP-Hackathon

# 🎬 HyperClipper: Your AI Video Librarian 🤖

Tired of scrubbing through hours of video to find that *one* perfect moment? HyperClipper is your personal AI video librarian that watches, understands, and catalogs your entire video library, making every second instantly searchable.

<br>

## ✨ What It Does

HyperClipper ingests your videos and uses a sophisticated AI pipeline to automatically:

-   **👁️ Watch Everything:** It analyzes video frames to detect scene changes and logical breaks.
-   **👂 Listen to Conversations:** It transcribes all spoken dialogue with pinpoint accuracy.
-   **🧠 Understand Content:** It combines visual and audio context to generate rich, meaningful summaries for each detected clip.
-   **🗂️ Create a Smart Index:** It stores everything in a high-speed vector database, turning your video content into a searchable knowledge base.

The result? A powerful, conversational AI agent you can chat with to find exactly what you're looking for, instantly.

<br>

## 🚀 Key Features

*   **Conversational Search:** Just ask! "Find clips where they discuss Qwen models" or "Show me the part about cooking pasta."
*   **Multimodal Understanding:** The agent doesn't just search text; it understands the *content* of the clips, leveraging both visuals and audio.
*   **Instant Previews:** See search results with summaries, relevance scores, and playable video clips directly in the chat.
*   **Drop-and-Go Analysis:** Simply upload a new video, and HyperClipper automatically processes and adds it to your searchable library.

<br>

## 🛠️ Getting Started

Follow these steps to set up and run HyperClipper on your local machine.

### Step 1: System Dependencies

This project requires `ffmpeg` for video and audio processing. Install it using your system's package manager.

*   **On macOS (using Homebrew):**
    ```bash
    brew install ffmpeg
    ```
*   **On Debian/Ubuntu:**
    ```bash
    sudo apt-get update && sudo apt-get install ffmpeg
    ```

### Step 2: Clone the Repository

```bash
git clone https://github.com/HaithamIsmail/HyperClipper
cd HyperClipper
```

### Step 3: Install Python Packages

Create a virtual environment and install the required Python libraries.

```bash
# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

Create a `.env` file in the root of the project directory and add your API keys and model endpoints. Your `config.py` will load these variables.

```env
# .env file
HF_API_KEY="hf_..."
NEBIUS_API_KEY="your_nebius_key"
MODAL_VLM_URL="https://your-modal-endpoint-url..."
# ... add any other secrets or configuration variables here
```

### Step 5: Launch the Application

With all dependencies installed and configurations set, launch the Gradio web server.

```bash
gradio app.py
```
You should see a message like `🚀 Starting Video Search Agent...` and a local URL (e.g., `http://127.0.0.1:7860`). Open this URL in your browser.

### Step 6: Analyze & Search

1.  Navigate to the **"Video Analyzer"** tab in the web UI.
2.  Upload your first video file and click "Analyze Video".
3.  Once processing is complete, go to the **"Chat with Clips"** tab and start a conversation with your new AI video librarian!

---

**HyperClipper: Don't just watch your videos. Understand them.**