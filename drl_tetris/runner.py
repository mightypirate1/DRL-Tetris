from abc import ABC, abstractmethod
import signal
import logging

from drl_tetris.training_state.training_state import training_state

logger = logging.getLogger(__name__)
signals = {
    signal.SIGINT: 'SIGINT',
    signal.SIGTERM: 'SIGTERM'
}

class runner(ABC):
    @abstractmethod
    def __init__(self, settings, me=""):
        self.settings = settings
        self.training_state = training_state(me=me)
        signal.signal(signal.SIGINT, self.store_runner_state_and_exit)
        signal.signal(signal.SIGTERM, self.store_runner_state_and_exit)
        self.read_settings()
        if not self.recover_runner_state():
            self.create_runner_state()

    @abstractmethod
    def read_settings(self):
        pass

    @abstractmethod
    def create_runner_state(self):
        pass

    @abstractmethod
    def get_runner_state(self):
        pass

    @abstractmethod
    def set_runner_state(self):
        pass

    @abstractmethod
    def run(self):
        pass

    def store_runner_state_and_exit(self, signum, frame):
        logger.info(f"{self.training_state.me} Saving runner-state due to {signals[signum]}")
        self.training_state.alive_flag.unset()
        self.training_state.runner_state.set(
            self.get_runner_state()
        )

    def recover_runner_state(self):
        found_state, state = self.training_state.runner_state.get()
        if found_state:
            self.set_runner_state(state)
        else:
            logger.info(f"No runner-state found - starting from scratch!")
        return found_state