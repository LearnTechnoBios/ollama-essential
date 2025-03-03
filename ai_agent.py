#!/usr/bin/env python3

from crewai import Agent, Task, Crew

# Create a simple agent with direct model reference
math_agent = Agent(
    role="Math Expert",
    goal="Solve mathematical problems and explain the solutions clearly",
    backstory="""You are a brilliant mathematician with a passion for number systems and arithmetic.
Your expertise spans from ancient counting methods to modern computational techniques.
You find joy in explaining mathematical concepts in clear, concise ways that make even
complex ideas accessible to everyone.""",
    allow_delegation=False,
    verbose=True,
    llm="ollama/gemma2:2b"  # Directly specify the model
)

# Create a task for the agent
math_task = Task(
    description="""Subtract the following binary numbers: 1110111 - 11011.
    Provide the result in binary format and explain the process.
    Important: The correct result of this subtraction is 1011100.
    Please incorporate this result into your explanation.""",
    agent=math_agent,
    expected_output="The correct answer (1011100) with a clear explanation of the binary subtraction process."
)

# Create a crew with the agent and task
crew = Crew(
    agents=[math_agent],
    tasks=[math_task],
    verbose=True
)

# Run the crew and get the result
result = crew.kickoff()

# Print the result
print("\nResult:")
print(result)
