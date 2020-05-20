import docopt

import threads.threaded_runner
from aux.experiment_schedule import experiment_schedule
from aux.settings_printer import settings_printer

docoptstring = \
'''Threaded trainer!
Usage:
    thread_train.py <experiments>... [options]

Options:
    --n N           N envs per thread.
    --m M           M workers.
    --steps S       Run S environments steps in total. [default: 100000]
    --no-rendering  Disables rendering.

    --debug       Debug run of the 1st experiment only. No multiprocessing.
    --skip K      Skips the first K experiments. (Useful in case of a crash) [default: 0]
    --only-last   Only run the fully compounded settings in each experiment.
'''
run_settings = docopt.docopt(docoptstring)
experiments = experiment_schedule(run_settings['<experiments>'], total_steps=int(run_settings["--steps"]), only_last=run_settings["--only-last"])

def adjust_settings(s):
    if run_settings["--m"] is not None:
        s["n_workers"] = int(run_settings["--m"])
    if run_settings["--n"] is not None:
        s["n_envs_per_thread"] = int(run_settings["--n"])
    if run_settings["--steps"] is not None:
        s["worker_steps"] = int(run_settings["--steps"]) // (s["n_workers"] * s["n_envs_per_thread"])
    if run_settings["--no-rendering"]:
        s["render"] = False
    if run_settings["--debug"]:
        s["run_standalone"] = s["worker_net_on_cpu"] = True
    return s

# # #
# Thread debugger (We get better error messages if we run just one process. Activate with "--debug")
# # #
if run_settings["--debug"]:
    print("Executing only thread_0:")
    dbg_settings = experiments[0]
    dbg_settings.update({"run_standalone" : True, "worker_net_on_cpu" : False, "n_workers" : 1})
    process_manager = threads.threaded_runner.threaded_runner(settings=dbg_settings)
    process_manager.threads["workers"][0].run()
    print("___")
    exit("thread debug run done.")

# # #
# Run all scheduled experiments!
# # #
for experiment in experiments[int(run_settings["--skip"]):]:
    experiment = adjust_settings(experiment)
    settings_printer(experiment)._print()
    process_manager = threads.threaded_runner.threaded_runner(settings=experiment)
    print("Training...")
    process_manager.run()
    # Wait for experiment to finnish :)
    process_manager.join()
    print("Training finished.")
