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

    # add the building dataset task
    print("[{}] {}".format(S[conf.BUILD_ENABLED], conf.BUILD_DESC))
    build = Step(conf.BUILD_SCRIPT, conf.BUILD_DESC, conf.BUILD_ARGS,
        conf.BUILD_ENABLED)
    pypeline.add(build)

    # add the training model task
    print("[{}] {}".format(S[conf.TRAIN_ENABLED], conf.TRAIN_DESC))
    train = Step(conf.TRAIN_SCRIPT, conf.TRAIN_DESC, conf.TRAIN_ARGS,
        conf.TRAIN_ENABLED)
    pypeline.add(train)

    # execute the pypeline
    print("[EXEC] executing pipeline: {}".format(pypeline))
    pypeline.execute()

if __name__ == "__main__":
    main()
