import click

from ssd import train, inference

@click.group()
def ssd_cli(): 
    pass

ssd_cli.add_command(train.train, 'train')
ssd_cli.add_command(inference.inference, 'inference')


if __name__ == "__main__":
    ssd_cli()
