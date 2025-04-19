import streamlit as st # [To Start] streamlit run dashboard.py
import pandas as pd

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

    # Combine recipe details into a single response string
    #response = recipe["name"] + " " + recipe["description"] + " ".join(recipe["ingredients"]) + " " + recipe["steps"]


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