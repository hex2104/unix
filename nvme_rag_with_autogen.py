# ----------------------------
# SECTION 1: Install Dependencies
# ----------------------------
# !pip install langchain faiss-cpu llama-cpp-python huggingface_hub sentence-transformers pymupdf autogen

# ----------------------------
# SECTION 2: Load and Split PDF
# ----------------------------
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

pdf_path = "data/nvme_spec_commands.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = splitter.split_documents(pages)
# Filter only Format NVM related chunks
format_nvm_docs = [d for d in docs if "Format NVM" in d.page_content]

# ----------------------------
# SECTION 3: Create and Save Vector Store with Local Embeddings
# ----------------------------
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(format_nvm_docs, embedding_model)
vectorstore.save_local("embeddings/faiss_index")

# ----------------------------
# SECTION 4: Load LLaMA 3.2 Locally via LlamaCpp
# ----------------------------
from langchain.llms import LlamaCpp

llm = LlamaCpp(
    model_path="models/llama-3.2-8B.Q4_K_M.gguf",  # Replace with your actual local model path
    temperature=0.1,
    max_tokens=2048,
    n_ctx=2048,
    n_batch=256,
    verbose=True
)

# ----------------------------
# SECTION 5: Setup Retrieval QA Chain
# ----------------------------
retriever = FAISS.load_local("embeddings/faiss_index", embedding_model).as_retriever()
from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ----------------------------
# SECTION 6: Add AutoGen Integration
# ----------------------------
import autogen
from autogen.agentchat.assistant import AssistantAgent
from autogen.agentchat.user_proxy import UserProxyAgent

# Create a wrapper function for our RAG system
def rag_function(query):
    """Function to query our RAG system about NVMe commands"""
    try:
        result = qa_chain.run(query)
        return f"RAG Result: {result}"
    except Exception as e:
        return f"Error querying RAG system: {str(e)}"

# Configure LLM for AutoGen
# For local LLM setups, we'll create a custom LLM config
llm_config = {
    "config_list": [
        {
            "model": "llama-3.2",  # Just a label for our local model
        }
    ],
    # This is important - we need to use a custom function to generate responses
    "temperature": 0.1,
    "timeout": 120,
    "cache_seed": 42,
}

# Create our agents
# 1. Technical Expert Agent that understands NVMe spec
nvme_expert = AssistantAgent(
    name="NVMe_Expert",
    system_message="You are an expert in NVMe specifications, particularly the Format NVM command. Answer technical questions with precision.",
    llm_config=llm_config,
)

# 2. Test Engineer Agent that generates test cases
test_engineer = AssistantAgent(
    name="Test_Engineer",
    system_message="You are a skilled test engineer. Generate comprehensive test cases for NVMe commands based on specifications.",
    llm_config=llm_config,
)

# 3. User Proxy Agent that mediates between the human and other agents
user_proxy = UserProxyAgent(
    name="User_Proxy",
    human_input_mode="TERMINATE",  # Set to "ALWAYS" if you want to approve each agent message
    max_consecutive_auto_reply=3,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={"use_docker": False},
    function_map={"query_nvme_rag": rag_function}
)

# Example usage of AutoGen multi-agent conversation
def run_nvme_autogen_conversation(query):
    print(f"\n--- Starting AutoGen Conversation for: {query} ---\n")
    
    # Start a group chat with both expert agents
    groupchat = autogen.GroupChat(agents=[user_proxy, nvme_expert, test_engineer], messages=[], max_round=6)
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)
    
    # First, fetch info from our RAG system
    rag_info = rag_function(query)
    
    # Initiate the chat with our query and RAG results
    user_proxy.initiate_chat(
        manager,
        message=f"""
        I need information about NVMe Format NVM command. Here's my specific question:
        {query}
        
        Here's some relevant information from our knowledge base:
        {rag_info}
        
        First, the NVMe Expert should provide technical details.
        Then, the Test Engineer should suggest test cases based on those details.
        """
    )
    
    print("\n--- AutoGen Conversation Completed ---\n")

# Example: Run different types of queries
print("Running AutoGen conversation for summary...")
run_nvme_autogen_conversation("Summarize the Format NVM command and its key fields.")

print("\nRunning AutoGen conversation for test cases...")
run_nvme_autogen_conversation("What are 3 important test cases for the Format NVM command?")

# ----------------------------
# SECTION 7: Custom LLM Config for AutoGen
# ----------------------------
# Create a custom LLM class to integrate our LlamaCpp model with AutoGen
from typing import Dict, List, Optional, Union, Callable, Any
import json

class CustomLLM:
    """Custom LLM interface for LlamaCpp to work with AutoGen"""
    
    def __init__(self, llm_instance):
        self.llm = llm_instance
    
    def create(self, 
              messages: List[Dict],
              model: Optional[str] = None,
              temperature: Optional[float] = None,
              max_tokens: Optional[int] = None,
              **kwargs) -> Dict:
        """Process messages and return a response mimicking OpenAI's format"""
        
        # Extract the prompt from messages
        prompt = self._messages_to_prompt(messages)
        
        # Call the LlamaCpp model
        response = self.llm(prompt, max_tokens=max_tokens or 512, temperature=temperature or 0.1)
        
        # Format response like OpenAI's API
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": response
                    }
                }
            ]
        }
    
    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """Convert a list of messages to a single prompt string"""
        prompt_parts = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}\n")
            elif role == "user":
                prompt_parts.append(f"Human: {content}\n")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}\n")
            else:
                prompt_parts.append(f"{role.capitalize()}: {content}\n")
        
        prompt_parts.append("Assistant: ")
        return "".join(prompt_parts)

# Create the LLM interface
custom_llm = CustomLLM(llm)

# Override the default LLM in AutoGen's config to use our custom implementation
autogen.ChatCompletion.register_provider(
    model_type="llama-3.2",
    create_func=custom_llm.create
)

# ----------------------------
# SECTION 8: Simple Interactive CLI for AutoGen Exploration
# ----------------------------
def interactive_cli():
    print("\n==== NVMe Spec Explorer with AutoGen ====")
    print("Enter 'exit' to quit")
    
    while True:
        user_input = input("\nEnter your question about NVMe Format NVM command: ")
        
        if user_input.lower() == 'exit':
            print("Exiting NVMe Explorer. Goodbye!")
            break
            
        # Choose interaction mode
        print("\nHow would you like to process this question?")
        print("1. Use simple RAG (LangChain only)")
        print("2. Use AutoGen multi-agent conversation")
        
        mode = input("Enter your choice (1 or 2): ")
        
        if mode == "1":
            # Simple RAG with LangChain
            try:
                result = qa_chain.run(user_input)
                print(f"\n--- LangChain RAG Answer ---\n{result}\n")
            except Exception as e:
                print(f"Error using RAG system: {str(e)}")
        elif mode == "2":
            # Multi-agent conversation with AutoGen
            try:
                run_nvme_autogen_conversation(user_input)
            except Exception as e:
                print(f"Error using AutoGen: {str(e)}")
                print("Autogen is still experimental with local models, you may need to modify the custom LLM implementation.")
        else:
            print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    interactive_cli()
