import click

from ssd import train

@click.group()
def ssd_cli(): 
    pass

ssd_cli.add_command(train.train, 'train')

if __name__ == "__main__":
    ssd_cli()
