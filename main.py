# ... (previous imports remain the same)
import streamlit as st
import sqlite3
import os
from datetime import datetime
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import uuid

# Initialize Stable Diffusion pipeline
@st.cache_resource
def load_model():
    try:
        from transformers import CLIPTextModel  # Test if transformers is installed
    except ImportError:
        st.error("The 'transformers' library is missing. Please install it with: pip install transformers")
        return None
    model_id = "runwayml/stable-diffusion-v1-5"
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, use_auth_token=os.getenv("HF_TOKEN")
        )
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        return pipe
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# Set up SQLite database
def init_db():
    conn = sqlite3.connect("prompts.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS prompts
                 (id TEXT, prompt TEXT, expected_style TEXT, image_path TEXT, timestamp TEXT)''')
    conn.commit()
    # Insert mock data if table is empty
    c.execute("SELECT COUNT(*) FROM prompts")
    if c.fetchone()[0] == 0:
        mock_data = [
            ("a fantasy castle in the clouds", "realistic"),
            ("a futuristic robot chef in a kitchen", "cyberpunk"),
            ("a panda riding a bicycle in space", "cartoon")
        ]
        for prompt, style in mock_data:
            c.execute("INSERT INTO prompts (id, prompt, expected_style, image_path, timestamp) VALUES (?, ?, ?, ?, ?)",
                      (str(uuid.uuid4()), prompt, style, "", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
    conn.close()

# Save prompt and image metadata to database
def save_prompt(prompt, style, image_path):
    conn = sqlite3.connect("prompts.db")
    c = conn.cursor()
    c.execute("INSERT INTO prompts (id, prompt, expected_style, image_path, timestamp) VALUES (?, ?, ?, ?, ?)",
              (str(uuid.uuid4()), prompt, style, image_path, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

# Get prompt history
def get_prompt_history():
    conn = sqlite3.connect("prompts.db")
    c = conn.cursor()
    c.execute("SELECT prompt, expected_style, image_path, timestamp FROM prompts")
    history = c.fetchall()
    conn.close()
    return history

# Generate evaluation report
def generate_report(history):
    report = "Evaluation Report\n" + "="*20 + "\n"
    for prompt, style, image_path, timestamp in history:
        if image_path:
            # Simple heuristic: check if style is mentioned in prompt or image exists
            alignment = "Aligned" if style.lower() in prompt.lower() else "Not fully aligned"
            report += f"Prompt: {prompt}\nStyle: {style}\nImage: {image_path}\nAlignment: {alignment}\nTimestamp: {timestamp}\n\n"
        else:
            report += f"Prompt: {prompt}\nStyle: {style}\nImage: Not generated\nAlignment: N/A\nTimestamp: {timestamp}\n\n"
    return report

# Delete prompt history and associated images
def delete_prompt_history():
    conn = sqlite3.connect("prompts.db")
    c = conn.cursor()
    c.execute("SELECT image_path FROM prompts WHERE image_path != ''")
    image_paths = [row[0] for row in c.fetchall()]
    
    # Delete images from filesystem
    for path in image_paths:
        if os.path.exists(path):
            os.remove(path)
    
    # Clear database table
    c.execute("DELETE FROM prompts")
    conn.commit()
    conn.close()


# Streamlit app
def main():
    st.title("Image Generator with Stable Diffusion")
    
    # Initialize database
    init_db()
    
    # Create directory for images
    if not os.path.exists("images"):
        os.makedirs("images")
    
    # Input form
    st.header("Generate Image")
    prompt = st.text_input("Enter your prompt:", placeholder="e.g., A dragon flying over mountains")
    style = st.selectbox("Expected Style:", ["realistic", "cyberpunk", "cartoon"])
    generate_button = st.button("Generate Image")
    
    if generate_button and prompt:
        with st.spinner("Generating image..."):
            try:
                pipe = load_model()
                image = pipe(prompt).images[0]
                image_path = f"images/{str(uuid.uuid4())}.png"
                image.save(image_path)
                save_prompt(prompt, style, image_path)
                st.image(image, caption=f"Generated: {prompt} ({style})")
            except Exception as e:
                st.error(f"Error generating image: {e}")
    
    # Prompt history
    st.header("Prompt History")
    history = get_prompt_history()
    if history:
        for prompt, style, image_path, timestamp in history:
            st.write(f"**Prompt**: {prompt} | **Style**: {style} | **Time**: {timestamp}")
            if image_path:
                st.image(image_path, width=200)
    else:
        st.write("No prompts yet.")
    
    # Image gallery
    st.header("Image Gallery")
    for prompt, style, image_path, timestamp in history:
        if image_path:
            st.image(image_path, caption=f"{prompt} ({style})", width=200)
    
    # Evaluation report
    st.header("Evaluation Report")
    if st.button("Generate Report"):
        report = generate_report(history)
        st.text_area("Report", report, height=300)
        st.download_button("Download Report", report, file_name="evaluation_report.txt")
# Delete history
    st.header("Delete History")
    if st.button("Clear All History"):
        delete_prompt_history()
        st.success("Prompt history and images have been deleted.")
if __name__ == "__main__":
    main()