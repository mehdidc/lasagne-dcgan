import click
from hyper import hyperjob
from train import traincollection
from train import dump

@click.group()
def main():
    pass

if __name__ == '__main__':
    main.add_command(traincollection)
    main.add_command(hyperjob)
    main.add_command(dump)
    main()
