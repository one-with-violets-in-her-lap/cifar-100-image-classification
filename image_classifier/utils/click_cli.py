import click


def start_click_cli_with_pretty_errors(entrypoint_command: click.BaseCommand):
    try:
        entrypoint_command()
    except Exception as error:
        click.echo(click.style(f"Error: {str(error)}", fg="red"))
