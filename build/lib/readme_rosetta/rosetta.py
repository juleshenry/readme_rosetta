

def main():
    # This is a backward compatibility shim or alternative entry point
    from .cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
