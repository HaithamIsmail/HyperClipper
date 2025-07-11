# pipeline_pytorch_optimized.py

import cv2
import os
import base64
import subprocess
import logging
from PIL import Image
import numpy as np
import pandas as pd
from queue import Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import time
from utils import get_image_embedding, get_text_embedding
from prompts import content_summary
from lancedb_utils import get_lancedb_table

from openai import OpenAI
from huggingface_hub import InferenceClient

from config import load_config

app_config = load_config()

# --- Configuration ---
class pipelineConfig:
    IMAGE_SIZE = (384, 384)
    SIMILARITY_THRESHOLD = 0.85
    VLM_MODEL = "Qwen/Qwen2.5-VL-32B-Instruct"
    WHISPER_MODEL_NAME = "whisper-1"
    
    FRAME_SAMPLING_RATE = 5  # Process every 5th frame for change detection.
    
    OUTPUT_DIR = "output"
    CSV_DIR = os.path.join(OUTPUT_DIR, "csvs")
    
    MAX_WORKERS_PROCESSES = os.cpu_count() or 1 # Default to 1 if cpu_count is not available
    MAX_WORKERS_THREADS = 10 # For I/O bound tasks

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
pipeline_config = pipelineConfig()

def get_modal_openai_client():
    return OpenAI(
        base_url=app_config.MODAL_VLM_URL.get_secret_value(),
        api_key=app_config.MODEL_API_KEY.get_secret_value()
    )

def get_hf_client():
    return InferenceClient(model=app_config.CLIP_MODEL_NAME, api_key=app_config.HF_API_KEY.get_secret_value())

# --- Utility and Database Functions ---
def create_directory(path):
    os.makedirs(path, exist_ok=True)

# --- Core Pipeline Functions ---

def _producer_read_frames(video_path, frame_queue, batch_size, sampling_rate):
    """
    (Producer Thread) Reads frames from the video file and puts them into a queue in batches.
    This is an I/O-bound task.
    """
    try:
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            logging.error(f"[Producer] Could not open video file: {video_path}")
            frame_queue.put(None) # Signal error/end
            return

        frame_index = 0
        while True:
            batch_cv2_frames = []
            while len(batch_cv2_frames) < batch_size:
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = video.read()
                if not ret:
                    # End of video
                    if batch_cv2_frames: # Put the last partial batch if it exists
                        frame_queue.put(batch_cv2_frames)
                    frame_queue.put(None) # Sentinel to signal the end
                    logging.info("[Producer] Reached end of video stream.")
                    video.release()
                    return
                
                batch_cv2_frames.append(frame)
                frame_index += sampling_rate

            frame_queue.put(batch_cv2_frames)
    except Exception as e:
        logging.error(f"[Producer] Error: {e}")
        frame_queue.put(None) # Ensure consumer doesn't block forever

def compare_frames_threaded_batched(video_path, batch_size=32, queue_size=2):
    """
    (Consumer) Computes frame similarities using a producer-consumer pattern.
    The main thread consumes batches for inference while a background
    thread produces the next batch by reading from the disk.
    """
    fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)

    frame_queue = Queue(maxsize=queue_size)
    producer_thread = Thread(
        target=_producer_read_frames,
        args=(video_path, frame_queue, batch_size, pipeline_config.FRAME_SAMPLING_RATE),
        daemon=True
    )
    producer_thread.start()
    logging.info("[Consumer] Producer thread started.")

    similarities = []
    last_frame_features = None
    
    while True:
        # 1. Get a pre-fetched batch from the queue (blocks until a batch is ready)
        batch_cv2_frames = frame_queue.get()
        
        # 2. Check for the sentinel value (end of stream)
        if batch_cv2_frames is None:
            logging.info("[Consumer] Received sentinel. Finishing processing.")
            break
        
        logging.info(f"[Consumer] Processing a batch of {len(batch_cv2_frames)} frames.")
        
        # 3. Preprocess the batch and convert to base64
        batch_images = []
        for frame in batch_cv2_frames:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            # Resize to model's expected size
            pil_image = pil_image.resize(pipeline_config.IMAGE_SIZE)
            # Convert to base64
            _, buffer = cv2.imencode(".jpg", cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR))
            batch_images.append(base64.b64encode(buffer).decode("utf-8"))
        
        # 4. Get features for the entire batch using HF Inference API
        batch_features = get_image_embedding(batch_images, app_config.CLIP_EMBEDDING_URL.get_secret_value())
        batch_features = np.array(batch_features)

        # 5. Compare with the last frame of the *previous* batch
        if last_frame_features is not None:
            # Calculate cosine similarity
            sim = np.dot(last_frame_features, batch_features[0]) / (
                np.linalg.norm(last_frame_features) * np.linalg.norm(batch_features[0])
            )
            similarities.append(float(sim))

        # 6. Calculate similarities *within* the current batch
        if len(batch_features) > 1:
            for i in range(len(batch_features) - 1):
                sim = np.dot(batch_features[i], batch_features[i + 1]) / (
                    np.linalg.norm(batch_features[i]) * np.linalg.norm(batch_features[i + 1])
                )
                similarities.append(float(sim))

        # 7. Store the features of the last frame for the next iteration
        last_frame_features = batch_features[-1]

    producer_thread.join() # Wait for the producer to finish cleanly
    logging.info(f"Calculated {len(similarities)} similarities using threaded batched processing.")
    return similarities, fps

def chunk_video(video_path):
    """
    Identifies clip boundaries based on frame similarity.
    (This function now calls the new batched version)
    """
    logging.info("Starting video chunking with batched GPU processing...")
    start_time = time.time()
    
    # ## CALL THE NEW BATCHED FUNCTION ##
    similarities, fps = compare_frames_threaded_batched(video_path)
    
    if not similarities:
        return [], [], [], []

    start_frames = [0]
    end_frames = []
    
    for i, score in enumerate(similarities):
        if score < pipeline_config.SIMILARITY_THRESHOLD:
            # Frame index is based on the sampling rate. The i-th similarity
            # is between frame (i * rate) and ((i+1) * rate).
            # The end of the scene is at frame ((i+1) * rate).
            frame_idx = (i + 1) * pipeline_config.FRAME_SAMPLING_RATE
            
            # Avoid creating tiny, meaningless clips
            if frame_idx > start_frames[-1] + 5*fps: # Clip must be at least 5 seconds long
                end_frames.append(frame_idx)
                start_frames.append(frame_idx) # The next clip starts where this one ended

    # The last clip goes to the end of the video
    total_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
    end_frames.append(total_frames - 1)

    start_times = [int(f / fps) for f in start_frames]
    # Ensure end time for the last clip is calculated properly or marked as -1
    # For simplicity, we'll calculate it from total_frames
    end_times = [int(f / fps) for f in end_frames]

    logging.info(f"Video chunking finished in {time.time() - start_time:.2f} seconds.")
    logging.info(f"Identified {len(start_frames)} potential clips.")
    
    return start_frames, end_frames, start_times, end_times

def process_clip(args):
    """
    ## OPTIMIZATION: This function is designed to be run in a separate process.
    It handles a single clip: extracts audio, the video segment, and thumbnail frames.
    """
    video_path, video_name, clip_idx, start_frame, end_frame = args
    
    try:
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        start_time_sec = start_frame / fps
        end_time_sec = (end_frame if end_frame != -1 else total_frames) / fps
        duration = end_time_sec - start_time_sec
        
        if duration <= 0:
            logging.warning(f"Clip {clip_idx} has zero or negative duration. Skipping.")
            return None

        output_dir = os.path.join(pipeline_config.OUTPUT_DIR, video_name, f"clip_{clip_idx}")
        create_directory(output_dir)
        
        audio_path = os.path.join(output_dir, "audio.mp3")
        clip_video_path = os.path.join(output_dir, "clip.mp4")

        # --- Audio Extraction using ffmpeg ---
        command_audio = [
            'ffmpeg', '-y', '-ss', str(start_time_sec), '-i', video_path, '-t', str(duration),
            '-q:a', '0', '-map', 'a', '-loglevel', 'error', audio_path
        ]
        subprocess.run(command_audio, check=True)
        
        # --- Video Clip Extraction using ffmpeg ---
        # Using 'copy' codec for speed, as it just remuxes without re-encoding
        command_video = [
            'ffmpeg', '-y', '-ss', str(start_time_sec), '-i', video_path, '-t', str(duration),
            '-c:v', 'copy', '-c:a', 'copy', '-loglevel', 'error', clip_video_path
        ]
        subprocess.run(command_video, check=True)

        # --- Thumbnail and Frame Extraction for VLM ---
        base64_frames = []
        frames_to_sample = 5
        frame_indices = np.linspace(start_frame, end_frame if end_frame != -1 else total_frames - 1, frames_to_sample, dtype=int)

        for idx in frame_indices:
            video.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = video.read()
            if success:
                _, buffer = cv2.imencode(".jpg", frame)
                base64_frames.append(base64.b64encode(buffer).decode("utf-8"))

        video.release()
        
        if not base64_frames:
             logging.warning(f"Could not extract any frames for clip {clip_idx}. Skipping.")
             return None
             
        return {
            "clip_idx": clip_idx,
            "base64_frames": base64_frames,
            "audio_path": audio_path,
            "clip_path": clip_video_path,
            "thumbnail": base64_frames[0]
        }
    except Exception as e:
        logging.error(f"Failed to process clip {clip_idx} for {video_name}: {e}")
        return None

def summarize_clip(clip_data):
    """
    ## OPTIMIZATION: This function is designed to be run in a thread.
    Takes clip data, transcribes audio, and generates a summary with a VLM.
    """
    transcription_client = OpenAI()
    client = get_modal_openai_client()
    try:
        # Transcribe audio
        with open(clip_data["audio_path"], "rb") as audio_file:
            transcription = transcription_client.audio.transcriptions.create(
                model=pipeline_config.WHISPER_MODEL_NAME,
                file=audio_file,
            ).text
        
        # Generate summary
        prompt = [
            {"role": "system", "content": content_summary},
            {"role": "user", "content": [
                {"type": "text", "text": f"These are frames from the video clip."},
                *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpeg;base64,{x}'}}, clip_data["base64_frames"]),
                {"type": "text", "text": f"The audio transcription is: {transcription}"}
            ]},
        ]
        
        response = client.chat.completions.create(
            model=pipeline_config.VLM_MODEL,
            messages=prompt,
            temperature=0.2,
        )
        summary = response.choices[0].message.content
        full_text = summary + "\n\n---Transcription---\n" + transcription
        
        clip_data["summary"] = full_text
        logging.info(f"Successfully summarized clip {clip_data['clip_idx']}.")
        return clip_data

    except Exception as e:
        logging.error(f"Failed to summarize clip {clip_data['clip_idx']}: {e}")
        clip_data["summary"] = "Error during summarization."
        return clip_data


def run_pipeline(video_path):
    """Main pipeline execution logic."""
    video_name = os.path.basename(video_path).split(".")[0]
    create_directory(pipeline_config.OUTPUT_DIR)
    create_directory(pipeline_config.CSV_DIR)
    create_directory(app_config.LANCEDB_URI.get_secret_value())

    # 1. Chunk video based on visual similarity
    start_frames, end_frames, start_times, end_times = chunk_video(video_path)
    if not start_frames:
        logging.error("No clips were generated. Exiting.")
        return

    # 2. Process clips in parallel (CPU-bound: audio/frame/video extraction)
    logging.info(f"Processing {len(start_frames)} clips using up to {pipeline_config.MAX_WORKERS_PROCESSES} processes...")
    processed_clip_data = []
    process_args = [(video_path, video_name, i, start, end) for i, (start, end) in enumerate(zip(start_frames, end_frames))]
    
    with ProcessPoolExecutor(max_workers=pipeline_config.MAX_WORKERS_PROCESSES) as executor:
        futures = [executor.submit(process_clip, arg) for arg in process_args]
        for future in as_completed(futures):
            result = future.result()
            if result:
                processed_clip_data.append(result)
    
    processed_clip_data.sort(key=lambda x: x['clip_idx'])
    
    # 3. Summarize clips in parallel (I/O-bound: API calls)
    logging.info(f"Summarizing {len(processed_clip_data)} clips using up to {pipeline_config.MAX_WORKERS_THREADS} threads...")
    summarized_results = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(summarize_clip, data) for data in processed_clip_data]
        for future in as_completed(futures):
            summarized_results.append(future.result())

    summarized_results.sort(key=lambda x: x['clip_idx'])
    
    # 4. Generate text embeddings for summaries (CPU/GPU-bound)
    texts_to_embed = [res['summary'] for res in summarized_results]
    embeddings = get_text_embedding(texts_to_embed, app_config.CLIP_EMBEDDING_URL.get_secret_value())

    # 5. Prepare and save data to CSV and LanceDB
    lancedb_data = []
    for i, result in enumerate(summarized_results):
        clip_idx = result['clip_idx']
        lancedb_data.append({
            'clip_id': clip_idx,
            'video_name': video_name,
            'clip_path': result['clip_path'],
            'start_time': start_times[clip_idx],
            'end_time': end_times[clip_idx],
            'summary': result['summary'],
            'thumbnail': result['thumbnail'],
            'vector': embeddings[i]
        })

    # Save to CSV as a backup/for analysis
    df = pd.DataFrame(lancedb_data)
    # The vector column can be large, so we might want to drop it for the CSV
    df.drop(columns=['vector']).to_csv(
        os.path.join(pipeline_config.CSV_DIR, f"{video_name}.csv"), index=False
    )
    logging.info(f"Results summary saved to {os.path.join(pipeline_config.CSV_DIR, f'{video_name}.csv')}")

    # Insert into LanceDB
    try:
        table = get_lancedb_table(app_config.LANCEDB_URI.get_secret_value(), app_config.CLIP_EMBEDDING_DIM)
        table.add(lancedb_data)
        logging.info(f"Successfully inserted {len(lancedb_data)} clips into LanceDB.")
    except Exception as e:
        logging.error(f"Failed to insert data into LanceDB: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimized Video Processing Pipeline with LanceDB.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the video file")
    
    args = parser.parse_args()
    
    total_start_time = time.time()
    run_pipeline(args.video_path)
    logging.info(f"Total pipeline execution time: {time.time() - total_start_time:.2f} seconds.")