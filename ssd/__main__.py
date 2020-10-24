import click

from ssd import train, inference, trace_module

@click.group()
def ssd_cli(): 
    pass

ssd_cli.add_command(train.train, 'train')
ssd_cli.add_command(inference.inference, 'inference')
ssd_cli.add_command(trace_module.trace_ssd, 'jit-serialize')



if __name__ == "__main__":
    ssd_cli()
