
import json

from langchain_community.document_loaders import TextLoader 
from langchain_text_splitters import CharacterTextSplitter 
from langchain_openai import OpenAIEmbeddings 
from langchain_chroma import Chroma
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
import pandas as pd

from langchain.schema import Document

# Tool 1: Search for a recipe in the Chroma database
def search_recipe(query: str) -> Document:
    # Load the recipe descriptions
    raw_documents = TextLoader("output_data/recipe_description.txt").load()
    text_splitter = CharacterTextSplitter(chunk_overlap=0, separator="\n") #Each line is a separate recipe
    documents = text_splitter.split_documents(raw_documents)

    # Indexing the chunks
    db_recipes = Chroma.from_documents(
    documents,
    embedding=OpenAIEmbeddings())
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
        "'name', 'description', 'ingredients' (as a list of strings), and 'steps' (as a list of strings)."
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
