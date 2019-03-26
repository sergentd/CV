# USAGE
# python builder.py

# the script will launch a list of scripts with relevant arguments
# in the context of building the age-gender-recognition application

# Firstly : build the datasets (verify the path in scripts/conf/builder_conf.py)
# Secondly : convert the datasets into records for mxnet
# Then : train the models
# Eventually : evaluate the models
# Total number of steps in the pypeline : 12

# to activate or deactive a task, turn the "enabled" state of the task to
# 'False' in the scripts/conf/builder_conf.py file
# each script is fully independent and can be launched outside of the scope
# of the pypeline with proper arguments
# this script purpose is to make a full pypeline to build a working application
# in a minimum of time (but with no guaranteed results as the learning rate
# is NOT updated automaticaly during the learning process)

# import necessary packages
from scripts.conf import builder_conf as conf
from helpers.pypeline import Pypeline
from helpers.pypeline import PyScript
from helpers.pypeline import ShellScript
from helpers.pypeline import Pause
from helpers.pypeline import Warning

def main():
    # printing symbols, for display purpose only
    S = {True:"x", False:" "}

    # set up the pypeline
    print("[PIPE] pypeline creation and feeding...")
    pypeline = Pypeline()

    # ask the pypeline to keep track of succeded task and disable it when
    # we re-run the script later (to avoid repeating the dataset buildings)
    # pypeline.disable_task()

    # add the building adience dataset tasks
    for (desc, args, enable) in zip(conf.BUILD_DESC, conf.BUILD_ARGS,
        conf.BUILD_ENABLED):
        print("[{}] {}".format(S[enable], desc))
        build = PyScript(conf.BUILD_SCRIPT, desc, args, enable)
        pypeline.add(build)

    # add a warning about mxnet
    pypeline.add(Warning("The next steps will produce records with "
    " the help of im2rec binary.\n"
    "Please be sure the path is correct in {}".format(conf.REC_SCRIPT)))
    pypeline.add(Pause())

    # loop over the records to create and add the tasks
    for (desc, args, enable) in zip(conf.REC_DESC, conf.REC_ARGS,
        conf.REC_ENABLED):
        print("[{}] {}".format(S[enable], desc))
        rec = ShellScript(conf.REC_SCRIPT, desc, args, enable)
        pypeline.add(rec)

    # add the training 'age' and 'gender' models tasks
    for (desc, args, enable) in zip(conf.TRAIN_DESC, conf.TRAIN_ARGS,
        conf.TRAIN_ENABLED):
        print("[{}] {}".format(S[enable], desc))
        train = PyScript(conf.TRAIN_SCRIPT, desc, args, enable)
        pypeline.add(train)

    # add the 'age' and 'gender' model evaluation task
    for (desc, args, enable) in zip(conf.EVAL_DESC, conf.EVAL_ARGS,
        conf.EVAL_ENABLED):
        print("[{}] {}".format(S[enable], desc))
        eval = PyScript(conf.EVAL_SCRIPT, desc, args, enable)
        pypeline.add(eval)

    # execute the pypeline
    print("[EXEC] executing pipeline: {} steps".format(len(
        [x for x in pypeline.steps if x.type not in ["warning", "pause"] and
            x.enable])))
    pypeline.execute()

if __name__ == "__main__":
    main()
