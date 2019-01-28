from ml import LearningAgent
from tensorflow import train, Session, trainable_variables
from os import path
from json import dumps

ckpt_dir = path.join("saves")
debg_path = path.join("./", "vals.txt")


def print_all():
    l_a = LearningAgent(2)
    saver = train.Saver()
    ckp = train.get_checkpoint_state(ckpt_dir)
    vls = {}
    with Session() as sess:
        saver.restore(sess, ckp.model_checkpoint_path)
        tr_vars = trainable_variables()
        for var in tr_vars:
            vls[str(var)] = sess.run(var).tolist()
    depg_file = open(debg_path, 'w')
    depg_file.write(dumps(vls))
    depg_file.flush()
    depg_file.close()


if __name__ == "__main__":
    print_all()
