import click
from hyper import runhyper, inserthyper
from train import traincollection
from train import dump

@click.group()
def main():
    pass

if __name__ == '__main__':
    main.add_command(traincollection)
    main.add_command(runhyper)
    main.add_command(inserthyper)
    main.add_command(dump)
    main()
