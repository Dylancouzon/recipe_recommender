import streamlit as st # [To Start] streamlit run dashboard.py

from dotenv import load_dotenv
load_dotenv()

from instruments import tracer
from tools import search_tool, generate_tool
from evaluators import recipe_relevance_evaluator



# Router prompt function
@tracer.agent(name="agent")
def start_agent(query: str) -> dict:

    # Step 1: Search for a recipe
    retrieved_recipe = search_tool.run(query)

    # Step 2: Verify the relevance of the recipe based on the query
    evaluation_result = recipe_relevance_evaluator(retrieved_recipe, query)

    # If the recipe is relevant, return it or else, generate a new one with ChatGPT
    if evaluation_result == "relevant":
        final_recipe = retrieved_recipe
    else:
        new_recipe = generate_tool.run(query)
        final_recipe = new_recipe
    
    return final_recipe


# Streamlit Dashboard
st.title("Recipe Recommender")

# Input query from the user
query = st.text_input("What are you in the mood for?", placeholder="e.g., A quick and easy high-protein dinner")

# Display results when the user submits a query
if query:
    st.write(f"Top recipe for query: '{query}'")
    top_recipe = start_agent(query)

    # Style the recipe like a cooking recipe
    st.markdown(f"""
    <div style="background-color: black; padding: 20px; border-radius: 10px; border: 1px solid #ddd;">
        <h2 style="color: #4CAF50; text-align: center;">{top_recipe['name']}</h2>
        <p><strong>Description:</strong> {top_recipe['description']}</p>
        <hr style="border: 1px solid #ddd;">
        <p><strong>Ingredients:</strong></p>
        <ul>
            {"".join([f"<li>{ingredient}</li>" for ingredient in eval(top_recipe['ingredients'])])}
        </ul>
        <p><strong>Instructions:</strong></p>
        <ol>
            {"".join([f"<li>{steps}</li>" for steps in eval(top_recipe['steps'])])}
        </ol>
    </div>
    """, unsafe_allow_html=True)
