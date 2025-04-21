from phoenix.otel import register

from openinference.instrumentation.langchain import LangChainInstrumentor

tracer_provider = register(
    project_name="recipe_recommender",
    auto_instrument=True
)
tracer = tracer_provider.get_tracer(__name__)

LangChainInstrumentor().instrument(tracer_provider=tracer_provider)