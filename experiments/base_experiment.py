######
## Usage:
##  Fill out settings. A lot of settings are mandatory, and if you don't
##  know what to do, just copy those from a functioning project.
##  If this file is passed to an experiment_scheduler (e.g by passing it
##  to thread_train.py), an experiment will be run using settings. Once
##  completed, settings will be "patched" by calling on it .update(s)
##  where s is the first dict in patches. This process is repeated until
##  all patches are applied.
##
##  Basically: specify settings for your project. To do batch-runs of
##  settings, you can apply patches. thread_train.py also works with any
##  number of experiment-files and patches. Just make sure you specify
##  a new run-id so you don't overwrite your data!
##
##  Note: Settings not specified reverts to their default values found
##  in aux/settings.py.
##
##  Note: when this code is expecuted, a variable named "total_steps" will
##  be available. It is an integer value specifying the total number of
##  steps that will be taken during training.

settings = {
            "run-id" : "base_0",
            }

patches = [
            {
                "run-id" : "base_1"
            },
            ]
