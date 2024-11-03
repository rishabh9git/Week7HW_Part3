import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

# Initialize the ChatGPT model
chat = ChatOpenAI(model="gpt-4", temperature=0.5)

### OpenAI Secret Key
my_secret_key = st.secrets['MyOpenAIKey']

# Prompt templates for each scenario
positive_experience_template = PromptTemplate.from_template(
    "The user had a positive experience. Respond professionally by thanking them for their feedback and for choosing our airline."
)
negative_airline_fault_template = PromptTemplate.from_template(
    "The user had a negative experience caused by the airline's fault (e.g., lost luggage). Respond professionally by offering sympathies and informing them that customer service will contact them soon to resolve the issue or provide compensation."
)
negative_beyond_control_template = PromptTemplate.from_template(
    "The user had a negative experience due to a reason beyond the airline's control (e.g., weather-related delay). Respond professionally by offering sympathies and explaining that the airline is not liable in such situations."
)

# Initialize chains for each prompt
positive_chain = LLMChain(llm=chat, prompt=positive_experience_template)
negative_airline_fault_chain = LLMChain(llm=chat, prompt=negative_airline_fault_template)
negative_beyond_control_chain = LLMChain(llm=chat, prompt=negative_beyond_control_template)

# Routing logic
def determine_experience_chain(user_input):
    if "lost luggage" in user_input or "airline" in user_input:
        return negative_airline_fault_chain
    elif "weather" in user_input or "uncontrollable" in user_input:
        return negative_beyond_control_chain
    elif "great" in user_input or "good" in user_input:
        return positive_chain
    else:
        return None

# Streamlit app setup
st.title("Airline Experience Feedback")

# User input section
user_input = st.text_area("Share with us your experience of the latest trip.")

if st.button("Submit"):
    chain_to_use = determine_experience_chain(user_input)
    
    # Run the appropriate chain based on routing logic
    if chain_to_use:
        response = chain_to_use.run(user_input=user_input)
        st.write("Response:", response)
    else:
        st.write("Thank you for your feedback! We appreciate you sharing your experience with us.")
