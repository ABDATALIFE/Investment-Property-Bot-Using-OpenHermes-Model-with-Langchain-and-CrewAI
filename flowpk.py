import os
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_community.llms import Ollama


#%%
os.environ["SERPER_API_KEY"] = "69af73d69fcc3dc1a06031dc347dfaf5412161fb"

llm = Ollama(model="openhermes")

search_tool = SerperDevTool()

#%%
researcher = Agent(
    llm=llm,
    role="Senior Property Researcher",
    goal="Find promising investment properties.",
    backstory="You are a veteran property analyst. In this case you're looking for retail properties to invest in.",
    allow_delegation=False,
    tools=[search_tool],
    verbose=True,
)


#%%
task1 = Task(
    description="Search the internet and find 5 promising real estate investment plots in Karachi, Pakistan. For each plot highlighting the demand price, area and any potential factors that would be useful to know for that area.",
    expected_output="""A detailed report of each of the Plot.The results should be formatted as shown below: 

    Plot 1: DHA Karachi
    Demand Price: PKR 1 Million
    Size: 1000 Yards
    Background Information: These plots are located in DHA Karachi . The following list highlights some of the top contenders for investment opportunities """,
    agent=researcher,
    output_file="task1_output.txt",
)


#%%

crew = Crew(agents=[researcher], tasks=[task1], verbose=2)

#%%
task_output = crew.kickoff()
#%%
print(task_output)

