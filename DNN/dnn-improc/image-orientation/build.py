# import necessary packages
from conf import config as conf
from pyimagesearch.pypeline import Pypeline
from pyimagesearch.pypeline import Step

S = {True:"+", False:"-"}

def main():
    # set up the pypeline
    print("[PIPE] pypeline creation and feeding:")
    pypeline = Pypeline()

    # add dataset creation task
    print("[{}] {}".format(S[conf.CREATE_ENABLED], conf.CREATE_DESC))
    create = Step(conf.CREATE_SCRIPT, conf.CREATE_DESC, conf.CREATE_ARGS,
        conf.CREATE_ENABLED)
    pypeline.add(extract)

    # add the features extraction task
    print("[{}] {}".format(S[conf.EXTRACT_ENABLED], conf.EXTRACT_DESC))
    extract = Step(conf.EXTRACT_SCRIPT, conf.EXTRACT_DESC, conf.EXTRACT_ARGS,
        conf.EXTRACT_ENABLED)
    pypeline.add(extract)

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
