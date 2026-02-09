from google.genai import types


def get_function_declarations():
    """Get function declarations for the Live API session."""
    start_background_task = types.FunctionDeclaration(
        name="start_background_task",
        description=(
            "Start a background task using the powerful reasoning agent. "
            "Use this when the user asks you to do something complex like "
            "research, analysis, writing, coding, or any task that would "
            "take more than a quick answer. The background agent can search "
            "the web and execute code."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "task_description": {
                    "type": "string",
                    "description": "Detailed description of what needs to be done",
                },
                "context": {
                    "type": "string",
                    "description": "Relevant context from the conversation or screen that would help complete the task",
                },
            },
            "required": ["task_description"],
        },
    )

    return [types.Tool(function_declarations=[start_background_task])]
