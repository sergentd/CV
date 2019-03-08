# import necessary packages
from conf import config as conf
from helpers.pypeline import Pypeline
from helpers.pypeline import Step

S = {True:"+", False:"-"}

def main():
    # set up the pypeline
    print("[PIPE] pypeline creation and feeding:")
    pypeline = Pypeline()

    # add the finding images containing faces task
    print("[{}] extract faces embeddings task".format(S[conf.EXTRACT_ENABLED]))
    find = Step(conf.EXTRACT_SCRIPT, conf.EXTRACT_DESC, conf.EXTRACT_ARGS,
        conf.EXTRACT_ENABLED)
    pypeline.add(find)

    # add the training model task
    print("[{}] train model task".format(S[conf.TRAIN_ENABLED]))
    encode = Step(conf.TRAIN_SCRIPT, conf.TRAIN_DESC, conf.TRAIN_ARGS,
        conf.TRAIN_ENABLED)
    pypeline.add(encode)

    # execute the pypeline
    print("[EXEC] executing pipeline: {}".format(pypeline))
    pypeline.execute()

if __name__ == "__main__":
    main()
