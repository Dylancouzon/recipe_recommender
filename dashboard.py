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

# Streamlit Dashboard
st.title("Recipe Recommender Dashboard")
st.write("Enter a query to retrieve the top 5 recipes based on semantic similarity.")

# Input query from the user
query = st.text_input("Enter your query:", placeholder="e.g., A quick and easy dinner")

# Display results when the user submits a query
if query:
    st.write(f"Top 5 recipes for query: '{query}'")
    top_recipes = retrieve_top_5_recipes(query)
    st.table(top_recipes)