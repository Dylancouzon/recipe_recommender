import pandas as pd
from langchain.schema import Document

import phoenix as px
from phoenix.trace import SpanEvaluations
from phoenix.evals import OpenAIModel, llm_classify
from opentelemetry.trace import format_span_id

from instruments import tracer

# Function to evaluate the relevance of the final recipe
def llm_eval_relevance(query: str, recipe) -> str:
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
    
    # Evaluate relevance of the recipe
    evaluation_result = llm_classify(
        data=pd.DataFrame([{"query": query, "recipe_name": recipe["name"], "recipe_description": recipe["description"], "recipe_ingredients": recipe["ingredients"]}]),
        template=classification_prompt,
        rails=["relevant", "irrelevant"],
        provide_explanation=True,
        model=OpenAIModel(model="gpt-4.1")
    )

    return evaluation_result

def recipe_relevance_evaluator(retrieved_recipe: Document, query: str) -> str:

    # Start a new evaluation span for tracing
    with tracer.start_as_current_span(
        "Recipe-evaluator",
        openinference_span_kind="evaluator",
    ) as eval_span:
        # Evaluate the relevance of the recipe
        evaluation_result = llm_eval_relevance(query, retrieved_recipe)

        # Add evaluation attributes to the span
        eval_span.set_attribute("eval.label", evaluation_result["label"][0])
        eval_span.set_attribute("eval.explanation", evaluation_result["explanation"][0])

    # Log the evaluation result
    span_id = format_span_id(eval_span.get_span_context().span_id)
    score = 1 if evaluation_result["label"][0] == "relevant" else 0
    eval_data = {
        "span_id": span_id,
        "label": evaluation_result["label"][0],
        "score": score,
        "explanation": evaluation_result["explanation"][0],
    }

    # Create a DataFrame for logging & log the evaluation data
    df = pd.DataFrame([eval_data])
    px.Client().log_evaluations(
        SpanEvaluations(
            dataframe=df,
            eval_name="Dataset relevance",
        ),
    )
    
    return evaluation_result["label"][0]