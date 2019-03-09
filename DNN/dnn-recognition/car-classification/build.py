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
    print("[{}] {}".format(S[conf.BUILD_ENABLED], conf.BUILD_DESC))
    build = Step(conf.BUILD_SCRIPT, conf.BUILD_DESC, conf.BUILD_ARGS,
        conf.BUILD_ENABLED)
    pypeline.add(build)

    # add the model task
    print("[{}] {}".format(S[conf.TUNE_ENABLED], conf.TUNE_DESC))
    tune = Step(conf.TUNE_SCRIPT, conf.TUNE_DESC, conf.TUNE_ARGS,
        conf.TUNE_ENABLED)
    pypeline.add(tune)

    # add the model task
    print("[{}] {}".format(S[conf.EVAL_ENABLED], conf.EVAL_DESC))
    tune = Step(conf.EVAL_SCRIPT, conf.EVAL_DESC, conf.EVAL_ARGS,
        conf.EVAL_ENABLED)
    pypeline.add(tune)

    # execute the pypeline
    print("[EXEC] executing pipeline: {}".format(pypeline))
    pypeline.execute()

if __name__ == "__main__":
    main()
