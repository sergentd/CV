# import necessary packages
from scripts.conf import builder_conf as conf
from helpers import Pypeline
from helpers import PyScript

def main():
    # printing symbols, for display purpose only
    S = {True:"+", False:"-"}

    # set up the pypeline
    print("[PIPE] pypeline creation and feeding:")
    pypeline = Pypeline()
    #pypeline.disable_task()

    # add the building dataset task
    for (desc, args, enable) in zip(conf.BUILD_DESC, conf.BUILD_ARGS,
        conf.BUILD_ENABLED):
        print("[{}] {}".format(S[enable], desc))
        task = PyScript(conf.BUILD_SCRIPT, desc, args, enable)
        pypeline.add(task)

    # add the training model task
    print("[{}] {}".format(S[conf.TRAIN_ENABLED], conf.TRAIN_DESC))
    task = PyScript(conf.TRAIN_SCRIPT, conf.TRAIN_DESC, conf.TRAIN_ARGS,
        conf.TRAIN_ENABLED)
    pypeline.add(task)

    # execute the pypeline
    print("[EXEC] executing pipeline: {}".format(pypeline))
    pypeline.execute()

if __name__ == "__main__":
    main()
