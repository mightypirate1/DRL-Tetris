import click
import logging

from drl_tetris.trainer import trainer

# TODO: phase these imports out, or reconside/refactor them
import experiments.presets
from tools.experiment_schedule import experiment_schedule
from tools.settings_printer import settings_printer
import tools.utils as utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@click.command()
@click.option("--experiment", default="experiments/sventon_ppo.py", help="experiment file")
def main(experiment):
    logger.info(f"Loading experiment from: {experiment}")
    experiments = experiment_schedule(
        experiment,
        only_last=True,
        overrides={"render": False},
    )
    for settings in experiments:
        logger.info(f"starting training with settings: {settings_printer(settings).format()}")
        trainer(utils.parse_settings(settings)).run()


if __name__ == "__main__":
    main()
