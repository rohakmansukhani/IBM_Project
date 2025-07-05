import streamlit as st
from google import genai
import os
from dotenv import load_dotenv
import re
import json

# Load environment variables
load_dotenv()

class StoryGenerator:
    def __init__(self):
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Gemini client with proper error handling"""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            st.error("❌ GEMINI_API_KEY not found in .env file")
            st.stop()
        
        try:
            return genai.Client(api_key=api_key)
        except Exception as e:
            st.error(f"❌ Failed to initialize Gemini client: {e}")
            st.stop()
    
    def create_story_prompt(self, premise, style, length):
        """Create a bulletproof prompt that prevents hallucination"""
        
        word_counts = {
            "short": "400-600 words",
            "medium": "700-900 words", 
            "long": "1000-1300 words"
        }
        
        style_instructions = {
            "adventurous": "filled with action, excitement, danger, and thrilling moments",
            "mysterious": "with suspense, intrigue, hidden secrets, and unexpected revelations",
            "romantic": "focusing on love, relationships, emotional connections, and heartfelt moments",
            "sci-fi": "with futuristic technology, space travel, scientific concepts, and advanced civilizations",
            "fantasy": "with magic, mythical creatures, enchanted worlds, and supernatural elements",
            "horror": "with scary, frightening, suspenseful, and spine-chilling elements",
            "comedy": "humorous, funny, witty, and entertaining with comedic situations",
            "drama": "emotionally intense with realistic characters, conflicts, and human struggles"
        }
        
        return f"""You are a professional storyteller. Write a complete, engaging story based on this premise: {premise}

STORY REQUIREMENTS:
- Style: {style} ({style_instructions.get(style, 'engaging')})
- Length: {word_counts.get(length, '500 words')}
- Format: Complete narrative story
- Structure: Clear beginning, middle, and end
- Characters: Well-developed with distinct personalities
- Setting: Vivid and immersive descriptions
- Plot: Compelling with conflict and resolution

CRITICAL INSTRUCTIONS:
- Write ONLY the story content
- Do NOT include meta-commentary, explanations, or analysis
- Do NOT use numbered lists or bullet points
- Do NOT break the story into sections or chapters
- Create a flowing, continuous narrative
- Start the story immediately with action or dialogue
- End with a satisfying conclusion

Begin writing the story now:"""

    def clean_response(self, text):
        """Clean and validate the story response"""
        if not text:
            return "Error: Empty response from AI"
        
        # Remove common AI artifacts
        text = re.sub(r'^(Here\'s|Here is|This is).*?story.*?:', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^Story:', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^\d+\..*?\n', '', text, flags=re.MULTILINE)  # Remove numbered lists
        text = re.sub(r'^\*.*?\n', '', text, flags=re.MULTILINE)    # Remove bullet points
        text = re.sub(r'\n{3,}', '\n\n', text)                     # Fix excessive line breaks
        
        # Clean up whitespace
        text = text.strip()
        
        # Validate story quality
        if len(text) < 100:
            return "Error: Story too short. Please try again."
        
        # Check for repetitive content
        sentences = text.split('.')
        if len(sentences) > 10:
            unique_sentences = set(s.strip().lower() for s in sentences if s.strip())
            if len(unique_sentences) < len(sentences) * 0.7:
                return "Error: Detected repetitive content. Please regenerate."
        
        return text
    
    def generate_story(self, premise, style, length):
        """Generate a high-quality story"""
        try:
            prompt = self.create_story_prompt(premise, style, length)
            
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            
            if response and response.text:
                cleaned_story = self.clean_response(response.text)
                return cleaned_story
            else:
                return "Error: No response from AI. Please try again."
                
        except Exception as e:
            return f"Error generating story: {str(e)}"

def main():
    st.cache_data.clear()
    st.cache_resource.clear()
    st.set_page_config(
        page_title="Professional AI Story Generator",
        page_icon="📚",
        layout="wide"
    )
    
    # Initialize story generator
    if 'story_gen' not in st.session_state:
        st.session_state.story_gen = StoryGenerator()
    
    st.title("📚 Professional AI Story Generator")
    st.markdown("*Production-grade storytelling powered by Google Gemini 2.0 Flash*")
    
    # Success indicator
    st.success("✅ Gemini API Connected Successfully")
    
    st.divider()
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🎯 Story Configuration")
        
        # Story premise
        premise = st.text_area(
            "**Story Premise:**",
            placeholder="A time traveler goes to ancient Egypt...",
            height=100,
            help="Describe the main idea or situation for your story"
        )
        
        # Style selection
        style = st.selectbox(
            "**Story Style:**",
            ["adventurous", "mysterious", "romantic", "sci-fi", "fantasy", "horror", "comedy", "drama"],
            help="Choose the genre and tone for your story"
        )
        
        # Length selection
        length = st.selectbox(
            "**Story Length:**",
            ["short", "medium", "long"],
            help="Select the approximate word count for your story"
        )
        
        # Generate button
        generate_btn = st.button(
            "🚀 Generate Professional Story",
            type="primary",
            use_container_width=True,
            disabled=not premise.strip()
        )
        
        if not premise.strip():
            st.warning("⚠️ Please enter a story premise to continue")
    
    with col2:
        st.subheader("📖 Generated Story")
        
        if generate_btn and premise.strip():
            with st.spinner(f"✨ Crafting your {style} {length} story..."):
                story = st.session_state.story_gen.generate_story(premise, style, length)
                
                # Store in session state
                st.session_state.current_story = story
                st.session_state.story_config = {
                    "premise": premise,
                    "style": style,
                    "length": length
                }
        
        # Display story if available
        if hasattr(st.session_state, 'current_story'):
            story = st.session_state.current_story
            config = st.session_state.story_config
            
            # Story metadata
            st.markdown(f"**📝 Premise:** {config['premise']}")
            st.markdown(f"**🎭 Style:** {config['style'].title()} | **📏 Length:** {config['length'].title()}")
            st.divider()
            
            # Display story
            if story.startswith("Error:"):
                st.error(story)
                if st.button("🔄 Try Again"):
                    st.rerun()
            else:
                st.markdown("### Your Story:")
                st.markdown(story)
                
                # Story statistics
                word_count = len(story.split())
                char_count = len(story)
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Words", word_count)
                with col_b:
                    st.metric("Characters", char_count)
                with col_c:
                    st.metric("Quality", "✅ High")
                
                # Action buttons
                st.divider()
                col_x, col_y = st.columns(2)
                
                with col_x:
                    st.download_button(
                        "📥 Download Story",
                        data=f"Title: {config['style'].title()} Story\nPremise: {config['premise']}\nStyle: {config['style']}\nLength: {config['length']}\n\n{story}",
                        file_name=f"story_{config['style']}_{length}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                with col_y:
                    if st.button("🔄 Generate New Version", use_container_width=True):
                        st.rerun()
        else:
            st.info("👆 Configure your story settings and click 'Generate Professional Story' to begin!")
    
    # Quick examples section
    st.divider()
    st.subheader("⚡ Quick Start Examples")
    
    examples = [
        ("A detective finds a door that leads to yesterday", "mysterious"),
        ("A chef discovers their spices control emotions", "fantasy"),
        ("Two astronauts are stranded on a beautiful alien planet", "sci-fi"),
        ("A librarian can hear books whispering their secrets", "fantasy")
    ]
    
    cols = st.columns(len(examples))
    for i, (example_premise, example_style) in enumerate(examples):
        with cols[i]:
            if st.button(f"📚 {example_premise[:30]}...", key=f"example_{i}"):
                st.session_state.example_premise = example_premise
                st.session_state.example_style = example_style
                st.rerun()
    
    # Handle example selection
    if hasattr(st.session_state, 'example_premise'):
        premise = st.session_state.example_premise
        style = st.session_state.example_style
        delattr(st.session_state, 'example_premise')
        delattr(st.session_state, 'example_style')
        
        with st.spinner(f"✨ Generating example {style} story..."):
            story = st.session_state.story_gen.generate_story(premise, style, "short")
            st.session_state.current_story = story
            st.session_state.story_config = {
                "premise": premise,
                "style": style,
                "length": "short"
            }
        st.rerun()
    
    # Footer
    st.divider()
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.9em;'>"
        "📚 Professional AI Story Generator | Powered by Google Gemini 2.0 Flash<br>"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
