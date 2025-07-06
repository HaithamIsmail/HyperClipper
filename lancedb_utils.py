import lancedb
import pyarrow as pa
import logging

def get_lancedb_table(lancedb_uri, embdding_dim, table_name="video_clips"):
    """Connects to LanceDB and creates/opens a table with the correct schema."""
    db = lancedb.connect(lancedb_uri)
    
    # Define the schema for our video clips table
    schema = pa.schema([
        pa.field("clip_id", pa.int32()),
        pa.field("video_name", pa.string()),
        pa.field("clip_path", pa.string()), # Path to the saved .mp4 clip file
        pa.field("start_time", pa.int32()),
        pa.field("end_time", pa.int32()),
        pa.field("summary", pa.string()),
        pa.field("thumbnail", pa.string()), # Base64 encoded thumbnail
        pa.field("vector", pa.list_(pa.float32(), embdding_dim))
    ])
    
    try:
        tbl = db.open_table(table_name)
        logging.info(f"Opened existing LanceDB table '{table_name}'.")
    except:
        tbl = db.create_table(table_name, schema=schema)
        logging.info(f"Created new LanceDB table '{table_name}'.")
        
    return tbl

def retreive_clip(clip_id: str, lancedb_uri: str):
    db = lancedb.connect(lancedb_uri)
    table = db.open_table("video_clips")
    arrow_table = table.to_pandas()
    clip_row = arrow_table[arrow_table["clip_id"] == int(clip_id)].iloc[0]
    return clip_row

