import os
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Backend Logic (for meta-llama/Llama-3.2-1B-Instruct)
class ChatbotBackend:
    def __init__(self):
        """Initialize the model and tokenizer."""
        self.model_name = "meta-llama/Llama-3.2-1B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, use_auth_token="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"
        )
        
        # Add a padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, use_auth_token="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"
        )
        self.model.resize_token_embeddings(len(self.tokenizer))  # Resize the embeddings to accommodate new tokens

    def generate_response(self, prompt):
        """Generate a response based on the input prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=512,  # Adjust as needed
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id,  # Use the new padding token ID
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Initialize the backend
backend = ChatbotBackend()

# Frontend Logic (Streamlit UI)
# Set Streamlit app title
st.title("Alita: AI Assistant")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display previous chat messages
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input prompt from the user
prompt = st.chat_input("How can I help you?")
if prompt:
    # Display user message
    with st.chat_message(name="user", avatar="👨‍🦰"):
        st.markdown(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    # Generate a response using the backend
    with st.chat_message(name="assistant", avatar="🤖"):
        message_placeholder = st.empty()
        try:
            backend_response = backend.generate_response(prompt)
        except Exception as e:
            backend_response = f"Error generating response: {e}"

        # Display assistant's response
        message_placeholder.markdown(backend_response)
        st.session_state["messages"].append({"role": "assistant", "content": backend_response})
