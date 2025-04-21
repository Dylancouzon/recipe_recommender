from instruments import tracer

# Format the final recipe for display in the Phoenix Dashboard
@tracer.chain(name="final_recipe")
def display_final_recipe(recipe):
    recipe_text = f"Recipe Name: {recipe['name']}\n\n"
    recipe_text += f"Description: {recipe['description']}\n\n"
    recipe_text += "Ingredients:\n"
    recipe_text += "\n".join([f"- {ingredient}" for ingredient in recipe["ingredients"]]) + "\n\n"
    recipe_text += "Instructions:\n"
    recipe_text += "\n".join([f"{i + 1}. {step}" for i, step in enumerate(recipe["steps"])]) + "\n"

    return recipe_text