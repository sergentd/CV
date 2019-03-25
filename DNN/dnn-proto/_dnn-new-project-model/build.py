# import necessary packages
from conf import config as conf
from helpers import Pypeline
from helpers import PyScript
from helpers import ShellScript
from helpers import Pause
from helpers import Warning

def main():
    # printing symbols, for display purpose only
    S = {True:"+", False:"-"}

    # set up the pypeline
    print("[PIPE] pypeline creation and feeding:")
    pypeline = Pypeline()
    pypeline.disable_task()

    # add the model task
    print("[{}] {}".format(S[conf.TASK_ENABLED], conf.TASK_DESC))
    task = PyScript(conf.TASK_SCRIPT, conf.TASK_DESC, conf.TASK_ARGS,
        conf.TASK_ENABLED)
    pypeline.add(task)

    pypeline.add(Warning("warning example...\nyou can put"
    " what you want here."))
    pypeline.add(Pause())

    # multi instance of same script example
    descs = ["shell 1", "shell 2"]
    args = [(1, 1), (2, 2)]

    # loop over the records to create and add the tasks
    for (i, args) in enumerate(args):
        shell = ShellScript("scripts/test.sh", descs[i], args, True)
        pypeline.add(shell)

    # execute the pypeline
    print("[EXEC] executing pipeline: {}".format(pypeline))
    pypeline.execute()

if __name__ == "__main__":
    main()
