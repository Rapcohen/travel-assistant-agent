from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from travel_assistant_agent.agent.graph import invoke_agent

load_dotenv()


def main():
    console = Console()
    counter = 0

    console.rule(style="cyan")
    console.print('AI: Hello! I am your travel assistant. How can I help you today?', style='bold white')

    try:
        while True:
            user_input = console.input("[bold green]You > [/]").strip()
            if not user_input:
                continue

            counter += 1
            ai_content = invoke_agent(user_input)
            renderable = Markdown(ai_content)
            console.print(
                Panel(
                    renderable,
                    title=f"[bold blue]AI[{counter}][/]",
                    border_style="blue",
                    expand=True,
                )
            )
    except (KeyboardInterrupt, EOFError):
        console.print("\n[bold cyan]Session ended[/]")
        console.rule(style="cyan")


if __name__ == '__main__':
    main()