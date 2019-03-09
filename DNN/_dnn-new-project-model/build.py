# import necessary packages
from conf import config as conf
from helpers import Pypeline
from helpers import Step

def main():
    # printing symbols, for display purpose only
    S = {True:"+", False:"-"}

    # set up the pypeline
    print("[PIPE] pypeline creation and feeding:")
    pypeline = Pypeline()

    # add the model task
    print("[{}] {}".format(S[conf.TASK_ENABLED], conf.TASK_DESC))
    task = Step(conf.TASK_SCRIPT, conf.TASK_DESC, conf.TASK_ARGS,
        conf.TASK_ENABLED)
    pypeline.add(task)

    # execute the pypeline
    print("[EXEC] executing pipeline: {}".format(pypeline))
    pypeline.execute()

if __name__ == "__main__":
    main()
