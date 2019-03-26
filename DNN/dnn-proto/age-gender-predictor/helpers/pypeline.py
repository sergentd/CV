# import necessary packages
from __future__ import print_function
import numpy as np
import datetime
import pickle
import json
import sh

class Pypeline:
    def __init__(self, steps=[], verbose=False):
        # store the steps and the verbose status
        self.steps = steps
        self.verbose = verbose

        # enable the pypeline by default and keep the tasks enabled as well
        # after completing it
        self.enable = True
        self.track_status = False
        self.sucesses = {}

    def add(self, step):
        # bind the step to this pypeline and add it to the list of steps
        step.add(self)
        self.steps.append(step)

    def step(self, script, name, args, enable=True, repeat=1, verbose=True):
        # create a step and add it to the list of steps
        s = Step(script, name, args, enable, self, repeat, verbose)
        self.steps.add(step)

    def execute(self):
        # execute each step in the pypeline
        for step in self.steps:
            # if the pipeline is still enable, execute the script
            if self.enable:
                # check to see if we need to track the task status
                if self.track_status:
                    # check to see if we saw this step before and need to
                    # skip it
                    if step.desc in self.successes:
                        step.enable = False

                        self._debug("skipped {} (cause: success in previous round)")

                    # execute the step and then write the status in the file
                    step.execute()
                    self.successes[step.desc] = True

                # otherwise, just execute the step whitout any tracking info
                else:
                    step.execute()

            # otherwise, kill the pypeline
            else:
                print("-- killing pipeline")
                break

        # check to see if we need to take care of tracking status
        if self.track_status:
            self._write_status()

    def skip(self, i=0):
        # find the step at the i index and disable it
        step = self.steps.get(i, None)
        if step is not None:
            step.enable = False

    def kill(self):
        # disable the pypeline
        self.enable = False

    def enable(self):
        # enable the pypeline
        self.enable = True

    def disable_task(self):
        self.track_status = True
        self._load_status()

    def _write_status(self):
        f = open(".last_run.json", "w")
        f.write(json.dumps(self.successes))
        f.close()

    def _load_status(self):
        try:
            self.successes = json.loads(open(".last_run.json", "r").read())
        except Exception as e:
            self.successes = {}

    def __repr__(self):
        return " -> ".join(step.desc for step in self.steps if step.enable and
            step.type != "warning")

    def _debug(self, msg, msgType="[INFO]"):
        if self.verbose:
            print("{} {} - {}".format(msgType, msg, datetime.datetime.now()))

class Step:
    def __init__(self, desc, pypeline, enable, verbose=True):
        self.type = "step"
        self.desc = desc
        self.pypeline = pypeline
        self.enable = enable
        self.verbose = verbose

    def add(self, pypeline):
        self.pypeline = pypeline

    def enable(self):
        self.enable = True

    def disable(self):
        self.enable = False

    def execute(self):
        if self.enable:
            pass

    def __repr__(self):
        return "{} (enable: {})".format(self.desc, self.enable)

    def _debug(self, msg, msgType="[INFO]"):
        if self.verbose:
            print("{} {} - {}".format(msgType, msg, datetime.datetime.now()))

class PyScript(Step):
    def __init__(self, script, desc, args, enable=True, pypeline=None,
        repeat=1, verbose=True):
        # store the script, the step name and the arguments the script needs
        # along the enable state of the step and the repeat parameter
        self.type = "pyscript"
        self.script = script
        self.desc = desc
        self.args = args
        self.enable = enable
        self.pypeline = pypeline
        self.repeat = repeat
        self.verbose = verbose

    def execute(self):
        # check to see if the step should be executed or if it has been disabled
        if self.enable:
            # parse the arguments for the script and create the shell command
            args = self._parse_args()
            cmd = "{}{}".format(self.script, args)
            exec = sh.Command("python")

            # try to execute the script with the command build previously
            try:
                exec(cmd.split(" "))
                self._debug("finished step {} ({})".format(self.desc,
                    self.script))

            # if an error occur, disable the pypeline
            except Exception as e:
                if self.pypeline is not None:
                    self.pypeline.kill()
                self._debug(msg=e, msgType="[ERROR]")

        # otherwise, just print a message about the skipped status of the step
        else:
            self._debug("skipping step {} ({})".format(self.desc, self.script))

    def _parse_args(self):
        # initialize the string representing the arguments
        args = ""

        # parse the arguments
        for argument in self.args:
            args += " --{} {}".format(argument[0], argument[1])

        # return the parsed arguments in a string format
        return args

    def __repr__(self):
        return "{} {} (enable: {})".format(self.desc, self.script,
            self.enable)

class ShellScript(Step):
    def __init__(self, script, desc, args, enable=True, pypeline=None,
        repeat=1, verbose=True):
        # store the script, the step name and the arguments the script needs
        # along the enable state of the step and the repeat parameter
        self.type = "shellscript"
        self.script = script
        self.desc = desc
        self.args = args
        self.enable = enable
        self.pypeline = pypeline
        self.repeat = repeat
        self.verbose = verbose

    def execute(self):
        if self.enable:
            # parse the arguments and create the shell command
            args = self._parse_args()
            cmd = "{}{}".format(self.script, args)
            exec = sh.Command("bash")

            # try to execute the script with the command build previously
            try:
                exec(cmd.split(" "))
                self._debug("finished step {} ({})".format(self.desc,
                    self.script))

            # if an error occur, disable the pypeline
            except Exception as e:
                if self.pypeline is not None:
                    self.pypeline.kill()
                self._debug(msg=e, msgType="[ERROR]")

        # otherwise, just print a message about the skipped status of the step
        else:
            self._debug("skipping step {} ({})".format(self.desc, self.script))

    def _parse_args(self):
        # initialize the string representing the arguments
        args = ""

        # loop over the arguments
        for argument in self.args:
            args += " {}".format(str(argument))

        # return the parsed arguments
        return args

    def __repr__(self):
        return "{} {} (enable: {})".format(self.desc, self.script,
            self.enable)

class Pause(Step):
    def __init__(self, desc="pause", pypeline=None, enable=True):
        self.type = "pause"
        self.desc = desc
        self.pypeline = pypeline
        self.enable = enable

    def execute(self):
        if self.enable:
            print("[INFO] hit enter to continue...")
            input()

    def __repr__(self):
        return "pausing the pypeline execution"

class Warning(Step):
    def __init__(self, warning, pypeline=None, enable=True):
        self.type = "warning"
        self.desc =  warning
        self.pypeline = pypeline
        self.enable = enable

    def execute(self):
        if self.enable:
            print("\n////////////////// * WARNING * //////////////////")
            print(self.desc)
            print("-------------------------------------------------")

    def __repr__(self):
        return "warning to user"
