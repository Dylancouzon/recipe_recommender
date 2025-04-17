import streamlit as st # [To Start] streamlit run dashboard.py
import pandas as pd
from phoenix.otel import register
from langchain_community.document_loaders import TextLoader # Importing a custom text loader for the recipe description
from langchain_text_splitters import CharacterTextSplitter #Splitting the text into smaller chunks
from langchain_openai import OpenAIEmbeddings # Importing OpenAI embeddings for vectorization
from langchain_chroma import Chroma #Vector database for storing the embeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import Tool, StructuredTool
from langchain.schema import Document
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Configure the Phoenix tracer
tracer_provider = register(
    project_name="recipe_recommender",
    auto_instrument=True
)
tracer = tracer_provider.get_tracer(__name__)

# Import the automatic instrumentor from OpenInference
from openinference.instrumentation.openai import OpenAIInstrumentor

# Finish automatic instrumentation
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

recipes = pd.read_csv("output_data/common_ingredients_recipes.csv")

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

# Initialize the OpenAI LLM for chat-based models
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
# Define the input schema for the verify tool
class VerifyRecipeInput(BaseModel):
    recipe: str
    query: str

# Tool 2: Verify if the recipe matches the user's request
@tracer.chain(name="verify_recipe")
def verify_recipe(recipe: str, query: str) -> str:
    prompt = PromptTemplate(
        input_variables=["recipe", "query"],
        template=(
            "You are a recipe verifier. The user requested: {query}. "
            "Here is the retrieved recipe: {recipe}. "
            "Does the recipe name match the user request? Respond with 'yes' or 'no' and explain why."
        )
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({"recipe": recipe, "query": query})

verify_tool = StructuredTool(
    name="Verify Recipe",
    func=verify_recipe,
    description="Verify if the retrieved recipe name matches the user's request.",
    args_schema=VerifyRecipeInput  # Specify the input schema
)

# Tool 3: Generate a new recipe if no match is found
@tracer.chain(name="generate_recipe")
def generate_recipe(query: str) -> str:
    prompt = PromptTemplate(
        input_variables=["query"],
        template=(
            "The user requested: {query}. Generate a recipe that matches this request. "
            "Include a name, description, ingredients, and instructions."
        )
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({"query": query})

generate_tool = Tool(
    name="Generate Recipe",
    func=generate_recipe,
    description="Generate a new recipe that matches the user's request. It must contain a name, description, ingredients, and instructions."
)

# Function to parse recipe text into structured format
@tracer.chain(name="parse_recipe")
def parse_recipe(recipe_text: str, default_name: str) -> dict:
    # Split the recipe text into sections
    lines = recipe_text.split("\n")
    name = lines[0] if lines else default_name
    description = lines[1] if len(lines) > 1 else "No description available."
    
    # Extract ingredients and instructions
    try:
        ingredients_start = lines.index("Ingredients:") + 1
        instructions_start = lines.index("Instructions:")
        ingredients = lines[ingredients_start:instructions_start]
        instructions = lines[instructions_start + 1:]
    except ValueError:
        # Fallback if "Ingredients:" or "Instructions:" are not found
        ingredients = ["No ingredients available."]
        instructions = ["No instructions available."]

    # Clean and format the ingredients and instructions
    ingredients = [ingredient.strip().replace("[", "").replace("]", "").replace("'", "") for ingredient in ingredients]
    instructions = " ".join([instruction.strip() for instruction in instructions])

    return {
        "name": name,
        "description": description,
        "ingredients": ingredients,
        "steps": instructions
    }

# Router Prompt: Combine the tools
@tracer.chain(name="router_prompt")
def router_prompt(query: str) -> dict:
    # Step 1: Search for a recipe
    retrieved_recipe = search_tool.run(query)

    # Step 2: Verify the recipe
    verification_result = verify_tool.run({"recipe": retrieved_recipe.page_content, "query": query})

    if "yes" in verification_result.lower():
       
        recipe_id = int(retrieved_recipe.page_content.split()[0].strip())
        return recipes[recipes["id"] == recipe_id].iloc[0] 
    else:
        # Step 3: Generate a new recipe
        new_recipe = generate_tool.run(query)
        # Parse the generated recipe into a structured format
        return parse_recipe(new_recipe, default_name="Generated Recipe")

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
            {"".join([f"<li>{ingredient}</li>" for ingredient in top_recipe['ingredients']])}
        </ul>
        <p><strong>Instructions:</strong></p>
        <p>{top_recipe['steps']}</p>
    </div>
    """, unsafe_allow_html=True)