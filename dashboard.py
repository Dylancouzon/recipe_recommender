import streamlit as st # [To Start] streamlit run dashboard.py
import pandas as pd
import json

from langchain_community.document_loaders import TextLoader # Importing a custom text loader for the recipe description
from langchain_text_splitters import CharacterTextSplitter #Splitting the text into smaller chunks
from langchain_openai import OpenAIEmbeddings # Importing OpenAI embeddings for vectorization
from langchain_chroma import Chroma #Vector database for storing the embeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import Tool, StructuredTool
from langchain.schema import Document

from langchain.chat_models import ChatOpenAI
import phoenix as px
from phoenix.trace import SpanEvaluations
from phoenix.trace.dsl import SpanQuery
from phoenix.otel import register
from phoenix.evals import (
    RAG_RELEVANCY_PROMPT_RAILS_MAP,
    RAG_RELEVANCY_PROMPT_TEMPLATE,
    OpenAIModel,
    download_benchmark_dataset,
    llm_classify,
)

from dotenv import load_dotenv
load_dotenv()

# Configure the Phoenix tracer
tracer_provider = register(
    project_name="recipe_recommender",
    auto_instrument=True
)
tracer = tracer_provider.get_tracer(__name__)

#Instantiate the text splitter
raw_documents = TextLoader("output_data/recipe_description.txt").load()
text_splitter = CharacterTextSplitter(chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_documents)


#Creating the document embeddings and storing them in a vector database
# Added persist_directory to store the embeddings
db_recipes = Chroma.from_documents(
    documents,
    embedding=OpenAIEmbeddings(),
    persist_directory="output_data")

# Initialize the OpenAI LLM 
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# Tool 1: Search for a recipe in the Chroma database
@tracer.chain(name="search_recipe")
def search_recipe(query: str) -> Document:
    top_doc = db_recipes.similarity_search(query, k=1)[0]
    return top_doc

search_tool = Tool(
    name="Search Recipe",
    func=search_recipe,
    description="Search for a recipe in the Chroma database based on the user's query."
)

# Tool 3: Generate a new recipe if no match is found
@tracer.chain(name="generate_recipe")
def generate_recipe(query: str) -> dict:
    prompt = PromptTemplate(
        input_variables=["query"],
        template=(
            "The user requested: {query}. Generate a recipe that matches this request. "
            "Return the recipe as a JSON object with the following fields: "
            "'name', 'description', 'ingredients' (as a list of strings), and 'instructions' (as a list of strings)."
        )
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run({"query": query})
    return json.loads(result)

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
@tracer.chain(name="llm_eval_qa_answer")
def llm_eval_qa_answer(query: str, recipe) -> str:
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
        Your response must be a single word, either "correct" or "incorrect",
        and should not contain any text or characters aside from that word.
        "correct" means that the question is correctly and fully answered by the answer.
        "incorrect" means that the question is not correctly or only partially answered by the
        answer."""
    )

    # Use llm_classify to evaluate relevance
    evaluation_result = llm_classify(
        data=pd.DataFrame([{"query": query, "recipe_name": recipe["name"], "recipe_description": recipe["description"], "recipe_ingredients": recipe["ingredients"]}]),
        template=classification_prompt,
        rails=["correct", "incorrect"],
        provide_explanation=True,
        concurrency=1,
        model=OpenAIModel(model="gpt-4.1")
    )

    # Log the evaluation result in Phoenix
    #px.Client().log_evaluations(SpanEvaluations(eval_name="QA_Answer", dataframe=evaluation_result))

    return evaluation_result

# Router prompt function
@tracer.chain(name="router_prompt")
def router_prompt(query: str) -> dict:
    # Step 1: Search for a recipe
    retrieved_recipe = search_tool.run(query)

    # Load the recipe list to get the recipe details fro the ID
    recipes = pd.read_csv("output_data/common_ingredients_recipes.csv")
    
    #Extract the ID fronm the retrieved recipe description
    recipe_id = int(retrieved_recipe.page_content.split()[0].strip())

    # Get the recipe details from the ID
    top_recipe = recipes[recipes["id"] == recipe_id].iloc[0] 

    # Step 2: Verify the recipe
    verification_result = llm_eval_qa_answer(query, top_recipe)
    if verification_result["label"][0] == "correct":
        # Return the recipe if it is relevant
        final_recipe = top_recipe
    else:
        # Step 3: Generate a new recipe
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