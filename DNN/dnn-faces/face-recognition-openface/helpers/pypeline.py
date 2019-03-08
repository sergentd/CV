# import necessary packages
from __future__ import print_function
import numpy as np
import datetime
import pickle
import sh

class Pypeline:
    def __init__(self, steps=[], verbose=False):
        # store the steps and the verbose status
        self.steps = steps
        self.verbose = verbose

        # enable the pypeline by default
        self.enable = True

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
                step.execute()

            # otherwise, kill the pypeline
            else:
                print("-- killing pipeline")
                break

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

    def __repr__(self):
        return " -> ".join(step.desc for step in self.steps if step.enable)

    def _debug(self, msg, msgType="[INFO]"):
        if self.verbose:
            print("{} {} - {}".format(msgType, msg, datetime.datetime.now()))

class Step:
    def __init__(self, script, desc, args, enable=True, pypeline=None,
        repeat=1, verbose=True):
        # store the script, the step name and the arguments the script needs
        # along the enable state of the step and the repeat parameter
        self.script = script
        self.desc = desc
        self.args = args
        self.enable = enable
        self.pypeline = pypeline
        self.repeat = repeat
        self.verbose = verbose

    def add(self, pypeline):
        # bind itself to a *unique* pypeline
        self.pipeline = pypeline

    def enable(self):
        # enable the step
        self.enable = True

    def disable(self):
        # disable the step
        self.enable = False

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

        # parse the argument which is expected to be formatting like
        # [["images", images],["arg2", value2]]
        for argument in self.args:
            args += " --{} {}".format(argument[0], argument[1])

        # return the parsed arguments in a string format
        return args

    def __repr__(self):
        return "{} {} (enable: {})".format(self.desc, self.script,
            self.enable)

    def _debug(self, msg, msgType="[INFO]"):
        if self.verbose:
            print("{} {} - {}".format(msgType, msg, datetime.datetime.now()))
