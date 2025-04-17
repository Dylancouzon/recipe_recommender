import streamlit as st # [Top Start] streamlit run dashboard.py
import pandas as pd
from phoenix.otel import register
from langchain_community.document_loaders import TextLoader # Importing a custom text loader for the recipe description
from langchain_text_splitters import CharacterTextSplitter #Splitting the text into smaller chunks
from langchain_openai import OpenAIEmbeddings # Importing OpenAI embeddings for vectorization
from langchain_chroma import Chroma #Vector database for storing the embeddings
from dotenv import load_dotenv

load_dotenv()

recipes = pd.read_csv("output_data/common_ingredients_recipes.csv")

#Instantiate the text splitter
raw_documents = TextLoader("output_data/recipe_description.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_documents)


#Creating the document embeddings and storing them in a vector database
# Added persist_directory to store the embeddings
db_recipes = Chroma.from_documents(
    documents,
    embedding=OpenAIEmbeddings(),
    persist_directory="output_data")

# Configure the Phoenix tracer
tracer_provider = register(
    project_name="recipe_recommender",
    auto_instrument=True
)

tracer = tracer_provider.get_tracer(__name__)

# Function to retrieve the top 5 recipes based on a query
@tracer.chain
def retrieve_top_5_recipes(query: str) -> pd.DataFrame:
    top_docs = db_recipes.similarity_search(query, k=5)
    recipe_ids = [int(doc.page_content.split()[0].strip()) for doc in top_docs]
    return recipes[recipes["id"].isin(recipe_ids)]

# Function to retrieve the top recipe based on a query
@tracer.chain
def retrieve_top_recipe(query: str) -> pd.Series:
    top_doc = db_recipes.similarity_search(query, k=1)[0]  # Retrieve only the top document
    recipe_id = int(top_doc.page_content.split()[0].strip())
    return recipes[recipes["id"] == recipe_id].iloc[0]  # Return the top recipe as a Series

# Streamlit Dashboard
st.title("Recipe Recommender Dashboard")
st.write("Enter a query to retrieve the top recipe based on semantic similarity.")

# Input query from the user
query = st.text_input("Enter your query:", placeholder="e.g., A quick and easy dinner")

# Display results when the user submits a query
if query:
    st.write(f"Top recipe for query: '{query}'")
    top_recipe = retrieve_top_recipe(query)

    # Style the recipe like a cooking recipe
    st.markdown(f"""
    <div style="background-color: black; padding: 20px; border-radius: 10px; border: 1px solid #ddd;">
        <h2 style="color: #4CAF50; text-align: center;">{top_recipe['name']}</h2>
        <p><strong>Description:</strong> {top_recipe['description']}</p>
        <hr style="border: 1px solid #ddd;">
        <p><strong>Ingredients:</strong></p>
        <ul>
            {"".join([f"<li>{ingredient.strip().replace('[', '').replace(']', '')}</li>" for ingredient in top_recipe['ingredients'].split(",")])}
        </ul>
        <p><strong>Instructions:</strong></p>
        <p>{top_recipe['description']}</p>
    </div>
    """, unsafe_allow_html=True)