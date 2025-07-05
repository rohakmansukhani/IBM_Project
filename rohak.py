import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers.pipelines import pipeline
import logging
import time
import torch
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Checking available devices...")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
if hasattr(torch, 'mps'):
    logger.info(f"MPS available: {torch.mps.is_available()}")
else:
    logger.info("MPS not available in this PyTorch version")

if hasattr(torch, 'mps') and torch.mps.is_available():
    device = 'mps'
    logger.info("Using MPS device")
elif torch.cuda.is_available():
    device = 'cuda'
    logger.info("Using CUDA device")
else:
    device = 'cpu'
    logger.info("Using CPU device")

@st.cache_resource
def load_story_generator():
    try:
        text_generator = pipeline(
            "text-generation",
            model="EleutherAI/gpt-neo-125M",
            device_map="auto" if device != "mps" else "cpu",
            max_new_tokens=300,
            do_sample=True,
            temperature=0.8,
            repetition_penalty=1.2,
            return_full_text=False
        )
        logger.info("GPT-Neo-125M story generation pipeline loaded successfully")
        
        llm = HuggingFacePipeline(pipeline=text_generator)
        logger.info("LangChain wrapper created successfully")
        return llm
    except Exception as e:
        logger.error(f"Error setting up GPT-Neo-125M pipeline: {e}")
        raise

story_prompt = PromptTemplate(
    input_variables=["prompt", "style", "length"],
    template="""
Write a {length} {style} story based on this prompt: {prompt}

Create an engaging story with:
- Interesting characters and dialogue
- A clear beginning, middle, and end
- Vivid descriptions and atmosphere
- A compelling plot with conflict and resolution

Story:
"""
)

def clean_generated_story(text):
    text = re.sub(r'\n+', '\n\n', text.strip())
    
    sentences = text.split('.')
    if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
        text = '.'.join(sentences[:-1]) + '.'
    
    text = re.sub(r'\.(\s*)([a-z])', lambda m: '.' + m.group(1) + m.group(2).upper(), text)
    
    return text

def generate_story(prompt, style="adventurous", length="short"):
    try:
        llm = load_story_generator()
        
        story_chain = story_prompt | llm
        
        # Fixed: Pass input as a single dictionary argument
        result = story_chain.invoke({
            "prompt": prompt,
            "style": style,
            "length": length
        })
        
        if isinstance(result, str):
            return clean_generated_story(result)
        else:
            return "Error: Could not generate story. Please try again."
            
    except Exception as e:
        logger.error(f"Error generating story: {e}")
        return f"Error generating story: {str(e)}. Please try again with a different prompt."

def main():
    st.set_page_config(
        page_title="AI Story Generator",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("🎭 AI Creative Story Generator")
    st.markdown("*Powered by GPT-Neo-125M*")
    st.markdown("---")
    
    with st.sidebar:
        st.header("Story Settings")
        
        style = st.selectbox(
            "Choose Story Style:",
            ["adventurous", "mysterious", "romantic", "sci-fi", "fantasy", "horror", "comedy", "drama"]
        )
        
        length = st.selectbox(
            "Choose Story Length:",
            ["short", "medium", "long"]
        )
        
        st.subheader("Tips for Better Stories")
        st.info("""
        - Be specific with your prompts
        - Include character details
        - Mention the setting
        - Add conflict or challenges
        
        **Example prompts:**
        - "A time traveler visits ancient Egypt"
        - "A detective discovers their reflection is missing"
        - "Two rival chefs compete in a magical cooking contest"
        """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("✍️ Enter Your Story Prompt")
        
        user_prompt = st.text_area(
            "What story would you like me to create?",
            placeholder="A time traveler visits ancient Egypt...",
            height=150,
            help="Describe the basic idea, characters, or setting for your story"
        )
        
        generate_button = st.button(
            "🚀 Generate Story",
            type="primary",
            use_container_width=True
        )
    
    with col2:
        st.subheader("📖 Generated Story")
        
        if hasattr(st.session_state, 'selected_prompt'):
            user_prompt = st.session_state.selected_prompt
            delattr(st.session_state, 'selected_prompt')
        
        if generate_button and user_prompt.strip():
            with st.spinner(f"Creating your {style} {length} story using GPT-Neo-125M..."):
                story = generate_story(user_prompt, style, length)
                
                st.markdown("### Your Story:")
                st.markdown(f"**Prompt:** {user_prompt}")
                st.markdown(f"**Style:** {style.title()} | **Length:** {length.title()}")
                st.markdown("---")
                st.write(story)
                
                st.download_button(
                    label="📥 Download Story",
                    data=f"Prompt: {user_prompt}\nStyle: {style}\nLength: {length}\n\n{story}",
                    file_name=f"ai_story_{int(time.time())}.txt",
                    mime="text/plain"
                )
        
        elif generate_button and not user_prompt.strip():
            st.warning("Please enter a story prompt first!")
        
        else:
            st.info("Enter a prompt and click 'Generate Story' to create your unique story!")
    
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center'>"
        "<p>Powered by GPT-Neo-125M & HuggingFace Transformers | Built with Streamlit</p>"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
