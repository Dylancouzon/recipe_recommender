import streamlit as st # [To Start] streamlit run main.py

from dotenv import load_dotenv
load_dotenv()

from instruments import tracer
from tools import search_tool, generate_tool
from evaluators import recipe_relevance_evaluator
from utils import display_final_recipe


# Starting the agent
# I experimented with chains & Langgraph to implement a router but it was making the code unecessarily complex
@tracer.agent(name="agent")
def start_agent(query: str) -> dict:

    # Step 1: Search for a recipe within the Chroma database
    retrieved_recipe = search_tool.run(query)

    # Step 2: Evaluate the relevance of the retrieved recipe
    evaluation_result = recipe_relevance_evaluator(retrieved_recipe, query)

    # If the recipe is relevant to the query, return it or else, generate a new one with ChatGPT
    if evaluation_result == "relevant":
        final_recipe = retrieved_recipe
        final_recipe['ingredients'] = eval(final_recipe['ingredients'])
        final_recipe['steps'] = eval(final_recipe['steps'])
    else:
        final_recipe = generate_tool.run(query)
    
    # Format & Log the final recipe in the Phoenix Dashboard
    display_final_recipe(final_recipe)
    return final_recipe


# Streamlit Dashboard
st.title("Recipe Recommender")

# Input query from the user
query = st.text_input("What are you in the mood for?", placeholder="e.g., A quick and easy high-protein dinner")

# Display results when the user submits a query
if query:
    top_recipe = start_agent(query)

    # Style the recipe like a cooking recipe
    st.markdown(f"""
    <div style="background-color: black; padding: 20px; border-radius: 10px; border: 1px solid #ddd;">
        <h2 style="color: #4CAF50; text-align: center;">{top_recipe['name']}</h2>
        <p><strong>Description:</strong> {top_recipe['description']}</p>
        <hr style="border: 1px solid #ddd;">
        <p><strong>Ingredients:</strong></p>
        <ul>
            {"".join([f"<li>{ingredient}</li>" for ingredient in top_recipe['ingredients']])}
        </ul>
        <p><strong>Instructions:</strong></p>
        <ol>
            {"".join([f"<li>{steps}</li>" for steps in top_recipe['steps']])}
        </ol>
    </div>
    """, unsafe_allow_html=True)
