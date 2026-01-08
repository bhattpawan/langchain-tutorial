from langchain_community.tools import DuckDuckGoSearchRun, ShellTool

duck_duck_go_tool = DuckDuckGoSearchRun()

shell_tool = ShellTool()

web_results = duck_duck_go_tool.run("Biggest news of today")

shell_results = shell_tool.run("echo Hello, World!")

print(shell_results)

