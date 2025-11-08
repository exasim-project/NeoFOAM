import typer


def hello():
    """Say hello from foamadapter CLI."""
    typer.echo("Hello, FoamAdapter user!")


app = typer.Typer()
app.command()(hello)


def main():
    app()


if __name__ == "__main__":
    main()
