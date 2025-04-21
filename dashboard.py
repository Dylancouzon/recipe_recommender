import streamlit as st # [To Start] streamlit run dashboard.py
import pandas as pd
import json

from langchain_community.document_loaders import TextLoader 
from langchain_text_splitters import CharacterTextSplitter 
from langchain_openai import OpenAIEmbeddings 
from langchain_chroma import Chroma
from langchain.tools import Tool
from langchain.schema import Document
from langchain_openai import ChatOpenAI

import phoenix as px
from phoenix.trace import SpanEvaluations
from phoenix.otel import register
from phoenix.evals import OpenAIModel, llm_classify

from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.trace import format_span_id

from dotenv import load_dotenv
load_dotenv()

# Configure the Phoenix tracer
tracer_provider = register(
    project_name="recipe_recommender",
    auto_instrument=True
)
tracer = tracer_provider.get_tracer(__name__)

LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

# Load the recipe descriptions
raw_documents = TextLoader("output_data/recipe_description.txt").load()
text_splitter = CharacterTextSplitter(chunk_overlap=0, separator="\n") #Each line is a separate recipe
documents = text_splitter.split_documents(raw_documents)

# Indexing the chunks
db_recipes = Chroma.from_documents(
    documents,
    embedding=OpenAIEmbeddings())


# Tool 1: Search for a recipe in the Chroma database
def search_recipe(query: str) -> Document:
    # Perform a similarity search in the Chroma database
    top_doc = db_recipes.similarity_search(query, k=1)[0]

    # Extract the recipe ID
    recipe_id = int(top_doc.page_content.split()[0].strip())

    # Retrieve & return the recipe details
    recipes = pd.read_csv("output_data/common_ingredients_recipes.csv")
    top_recipe = recipes[recipes["id"] == recipe_id].iloc[0] 

    return top_recipe

search_tool = Tool(
    name="Search Recipe",
    func=search_recipe,
    description="Search for a recipe in the Chroma database based on the user's query."
)

# Tool 2: Generate a new recipe if no match is found
def generate_recipe(query: str) -> dict:
    prompt = [
    (
        "system",
        "Generate a recipe that matches the user request. "
        "Return the recipe as a JSON object with the following fields: "
        "'name', 'description', 'ingredients' (as a list of strings), and 'instructions' (as a list of strings)."
    ),
    (query),
    ]
    llm = ChatOpenAI(model="gpt-4",temperature=0.7)
    result = llm.invoke(prompt)
    return json.loads(result.content)

generate_tool = Tool(
    name="Generate Recipe",
    func=generate_recipe,
    description="Generate a new recipe that matches the user's request. It must contain a name, description, ingredients, and instructions."
)

# Function to parse recipe text into structured format
@tracer.chain(name="parse_recipe")
def parse_recipe(recipe_json: dict, default_name: str) -> dict:
    # Extract fields from the JSON object
    name = recipe_json.get("name", default_name)
    description = recipe_json.get("description", "No description available.")
    ingredients = recipe_json.get("ingredients", ["No ingredients available."])
    instructions = recipe_json.get("instructions", ["No instructions available."])

    # Ensure ingredients and instructions are lists
    if not isinstance(ingredients, list):
        ingredients = [ingredients]
    if not isinstance(instructions, list):
        instructions = [instructions]

    return {
        "name": name,
        "description": description,
        "ingredients": str(ingredients),
        "steps": str(instructions)
    }

# Function to evaluate the relevance of the final recipe
def llm_eval_relevance(query: str, recipe) -> str:
    # Define the classification prompt
    classification_prompt = ("""You are given a query and a recipe. You must determine whether the
        given recipe is relevant to the query. Here is the data:
            [BEGIN DATA]
            ************
            [Query]: {query}
            ************
            [Recipe]: 
            Name: {recipe_name}
            Description: {recipe_description}
            Ingredients: {recipe_ingredients}
            [END DATA]
        Your response must be a single word, either "relevant" or "irrelevant",
        and should not contain any text or characters aside from that word.
        "correct" means that the question is correctly and fully answered by the answer.
        "incorrect" means that the question is not correctly or only partially answered by the
        answer."""
    )
    
    # Use llm_classify to evaluate relevance
    evaluation_result = llm_classify(
        data=pd.DataFrame([{"query": query, "recipe_name": recipe["name"], "recipe_description": recipe["description"], "recipe_ingredients": recipe["ingredients"]}]),
        template=classification_prompt,
        rails=["relevant", "irrelevant"],
        provide_explanation=True,
        model=OpenAIModel(model="gpt-4.1")
    )

    return evaluation_result

def recipe_relevance_evaluator(retrieved_recipe: Document) -> str:

    # Evaluation span for Recipe relevance
    with tracer.start_as_current_span(
        "Recipe-evaluator",
        openinference_span_kind="evaluator",
    ) as eval_span:
        evaluation_result = llm_eval_relevance(query, retrieved_recipe)
        eval_span.set_attribute("eval.label", evaluation_result["label"][0])
        eval_span.set_attribute("eval.explanation", evaluation_result["explanation"][0])

    # Logging our evaluation
    span_id = format_span_id(eval_span.get_span_context().span_id)
    score = 1 if evaluation_result["label"][0] == "relevant" else 0
    eval_data = {
        "span_id": span_id,
        "label": evaluation_result["label"][0],
        "score": score,
        "explanation": evaluation_result["explanation"][0],
    }
    df = pd.DataFrame([eval_data])
    px.Client().log_evaluations(
        SpanEvaluations(
            dataframe=df,
            eval_name="Dataset relevance",
        ),
    )
    
    return evaluation_result["label"][0]

# Router prompt function
@tracer.agent(name="agent")
def router_prompt(query: str) -> dict:

    # Step 1: Search for a recipe
    retrieved_recipe = search_tool.run(query)

    # Step 2: Verify the relevance of the recipe based on the query
    evaluation_result = recipe_relevance_evaluator(retrieved_recipe)

    # If the recipe is relevant, return it or else, generate a new one with ChatGPT
    if evaluation_result == "relevant":
        final_recipe = retrieved_recipe
    else:
        new_recipe = generate_tool.run(query)
        final_recipe = parse_recipe(new_recipe, "Generated Recipe")
    
    return final_recipe

# Streamlit Dashboard
st.title("Recipe Recommender")

# Input query from the user
query = st.text_input("What are you in the mood for?", placeholder="e.g., A quick and easy high-protein dinner")

# Display results when the user submits a query
if query:
    st.write(f"Top recipe for query: '{query}'")
    top_recipe = router_prompt(query)

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