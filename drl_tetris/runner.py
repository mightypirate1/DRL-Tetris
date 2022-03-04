from abc import ABC, abstractmethod
import signal
import logging

import hashlib
import dill

from drl_tetris.training_state.training_state import training_state

logger = logging.getLogger(__name__)
signals = {
    signal.SIGINT: 'SIGINT',
    signal.SIGTERM: 'SIGTERM'
}

#######
### runner:
#####
#
#   Base class for a stateful runner, implementing saving and retreiving of
#   runner state, and setting and retreiving of validation tokens that can
#   be used to verify that the recovered state matches expectations.
#
#####

class runner(ABC):
    @abstractmethod
    def __init__(self, settings, me=None):
        self.received_interrupt = False
        self.settings = settings
        self.training_state = training_state(me=me, scope=settings['run-id'])
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

    def graceful_exit(self):
        pass  # Say your last words

    def runner_validation(self, artifact):
        # Given an input, this is the function an aspiring restored runner needs to prove they can compute
        return artifact  # Proof of capability

    def validation_artifact(self):
        # Implement validation_artifact to return some object witch will be the input to the validation function runner_validation
        return 0  # Input to the proof of capability

    def store_runner_state_and_exit(self, signum, frame):
        logger.info(f"{self.training_state.me} Saving runner-state due to {signals[signum]}")
        self.training_state.alive_flag.unset()
        self.training_state.runner_state.set(
            self.get_runner_state()
        )
        validation_artifact = self.validation_artifact()
        self.set_validation_artifact(validation_artifact)
        logger.info(f"{self.training_state.me}: ARTIFACT CHECKSUM: [{self.compute_checksum(validation_artifact)}]")

        self.graceful_exit()
        self.received_interrupt = True

    def recover_runner_state(self):
        found_state, state = self.training_state.runner_state.get()
        if found_state:
            self.set_runner_state(state)
        else:
            logger.info(f"No runner-state found - starting from scratch!")
        return found_state

    def validate_runner(self):
        found, artifact, targetchecksum = self.get_validation_artifact()
        logger.info("--------------------------------")
        if not found:
            logger.info(self.training_state.validation_checksum)
            logger.info(f"no stored artifact")
            return True
        logger.info(f"target: [{targetchecksum}]")
        proof = self.runner_validation(artifact)
        checksum = self.compute_checksum(proof)

        logger.info(f"validation: [{checksum}]")
        if checksum == targetchecksum:  # If my result is equal to the stored target-result, it means I have successfully recovered runner-state
            return True
        raise RuntimeError(f"{self.training_state.me} unable to reproduce validation checksum ({checksum}) != {targetchecksum}")

    def set_validation_artifact(self, artifact):
        validation_target = self.runner_validation(artifact)  # Anyone who can perform this calculation correctly is good to take my place
        validation_checksum = self.compute_checksum(validation_target)
        self.training_state.validation_artifact.set(artifact)
        self.training_state.validation_checksum.set(validation_checksum)
        logger.info(self.training_state.validation_checksum)

    def get_validation_artifact(self):
        found_artifact, artifact = self.training_state.validation_artifact.get()
        found_checksum, checksum = self.training_state.validation_checksum.get()
        found = found_artifact and found_checksum
        return found, artifact, checksum

    def compute_checksum(self, artifact):
        return hashlib.md5(dill.dumps(artifact)).hexdigest()
