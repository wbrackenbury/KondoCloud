import tensorflow as tf
import numpy as np

from datasketch.minhash import _max_hash

from .utils import (MAX_SHARES, tokenize_fname, triplet_to_vec,
                    gen_token_universe)

VEC_SIZES = [1, 1, 128, 128, 2048, 871, 128, 1000]

LAST_MOD_MAX = 21.167
SIZE_MAX = 21.920

BASE = "/home/will/Documents/Research_2019/fm-interface/elfinder-flask/felfinder/swampnet/"
SAVE_PATH = BASE + "model_saves/swampnet/swampnet"
SHARE_SAVE_PATH = BASE + "model_saves/shares/shares"
FNAME_SAVE_PATH = BASE + "model_saves/fname/fname"
TREE_SAVE_PATH = BASE + "model_saves/tree/tree"


class RepContainer:

    def __init__(self, file_wrap, path_codes, toks, share_rep):

        self.lastmod_rep = np.array([np.log(file_wrap.last_modified)]) / LAST_MOD_MAX
        self.size_rep = np.array([np.log(file_wrap.size)]) / SIZE_MAX
        self.fname_rep = tokenize_fname(file_wrap.fname, toks)
        self.treepos_rep = triplet_to_vec(path_codes[file_wrap.path + '/'])

        self.content_rep = file_wrap.minhash_content.digest()
        self.content_rep = [a / _max_hash for a in self.content_rep]

        self.text_rep = file_wrap.w2v_rep
        self.im_rep = file_wrap.im_rep

        if self.text_rep is None:
            self.text_rep = np.random.normal(size=(128,)).astype('float32')

        if self.im_rep is None:
            self.im_rep = np.random.normal(size=(2048,)).astype('float32')
        else:
            self.im_rep = self.im_rep.reshape((2048,))

        # This gets inserted because we calculate these in bulk
        self.perms_rep = share_rep

        self.each_rep = (self.lastmod_rep,
                         self.size_rep,
                         self.content_rep,
                         self.text_rep,
                         self.im_rep,
                         self.fname_rep,
                         self.treepos_rep,
                         self.perms_rep)


def get_tree_model():

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(128,),
                              name='tree_dense_1'),
        tf.keras.layers.Dense(64, activation='relu', name='tree_dense_2'),
        tf.keras.layers.Dense(64, activation='relu', name='tree_dense_3'),
        tf.keras.layers.Dense(64, activation='relu', name='tree_dense_4'),
    ])

    model.load_weights(TREE_SAVE_PATH)

    return model


def get_fname_model():

    toks = gen_token_universe()
    d = len(toks)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(d,),
                              name='fname_dense_1'),
        tf.keras.layers.Dense(256, activation='relu',
                              name='fname_dense_2'),
        tf.keras.layers.Dense(128, activation='relu',
                              name='fname_dense_3'),
        tf.keras.layers.Dense(64, activation='relu',
                              name='fname_dense_4'),
    ])

    model.load_weights(FNAME_SAVE_PATH)

    return model


def get_share_model():

    d = MAX_SHARES

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, activation='relu', dtype='float64',
                              input_shape=(d,), name='shares_dense_1'),
        tf.keras.layers.Dense(256, activation='relu', dtype='float64',
                              name='shares_dense_2'),
        tf.keras.layers.Dense(128, activation='sigmoid', dtype='float64',
                              name='shares_dense_3'),
        tf.keras.layers.BatchNormalization(name='shares_batch_norm'),
        tf.keras.layers.Dropout(0.8),
        tf.keras.layers.Dense(64, activation='relu', dtype='float64', name='shares_out'),
    ])

    model.load_weights(SHARE_SAVE_PATH)

    return model


def swamp_model(train=False):

    """
    Prototype network architecture for SwampNet. Under development--not finalized.
    """

    lastmod_input = tf.keras.Input(shape=(1,), name='lastmod_input')
    size_input = tf.keras.Input(shape=(1,), name='size_input')
    content_input = tf.keras.Input(shape=(128,), name='content_input')

    fname_model = get_fname_model()
    treepos_model = get_tree_model()
    shares_model = get_share_model()

    fname_input = fname_model.layers[-1].output
    treepos_input = treepos_model.layers[-1].output
    shares_input = shares_model.layers[-1].output

    text_input = tf.keras.Input(shape=(128,), name='text_input')
    img_input = tf.keras.Input(shape=(2048,), name='img_input')

    inputs = [lastmod_input,
              size_input,
              content_input,
              text_input,
              img_input]

    concat_input = tf.concat(inputs, axis=1)
    upper_inputs = tf.keras.layers.concatenate([fname_input, treepos_input, shares_input])
    full_input = tf.keras.layers.concatenate([concat_input, upper_inputs], name='full_input')

    # SUM OF THE OUTPUTS
    layer_len = 2498

    multimodal_batch = tf.keras.layers.BatchNormalization()(full_input)
    multimodal_embed = tf.keras.layers.Dense(int(layer_len / 2),
                                             activation='relu')(multimodal_batch)
    multimodal_embed = tf.keras.layers.Dense(int(layer_len / 4),
                                             activation='relu')(multimodal_embed)

    multimodal_embed = tf.keras.layers.Dropout(0.5)(multimodal_embed)

    multimodal_embed = tf.keras.layers.Dense(int(layer_len / 8),
                                             activation='relu')(multimodal_embed)
    multimodal_embd = tf.keras.layers.Dropout(0.5)(multimodal_embed)

    multimodal_embed = tf.keras.layers.Dense(int(layer_len / 16),
                                             activation='relu')(multimodal_embed)
    multimodal_embed = tf.keras.layers.Dense(int(layer_len / 32),
                                             activation='relu')(multimodal_embed)

    full_model = tf.keras.models.Model(inputs=[lastmod_input,
                                               size_input,
                                               content_input,
                                               text_input,
                                               img_input,
                                               fname_model.input,
                                               treepos_model.input,
                                               shares_model.input],
                                       outputs=multimodal_embed)

    for layer in full_model.layers:
        if any([x in layer.name for x in ['tree', 'shares', 'fname']]):
            layer.trainable = False

    if not train:
        full_model.load_weights(SAVE_PATH)

    return full_model


def proc_full_reps(id_list, all_wraps, path_codes, toks, share_reps):

    """
    Take in the necessary pre-computed pre-requisites and
    return the model's vector representation for each wrapper
    """

    vec_rep_conts = {k: RepContainer(all_wraps[k], path_codes,
                                     toks, share_reps[i, :])
                     for i, k in enumerate(id_list)}

    joint_vec_reps = tf.concat([tf.reshape(
        tf.concat(vec_rep_conts[k].each_rep, axis=0),
        shape=(1, -1))
                                for k in id_list], axis=0)

    model = swamp_model(train=False)

    print(joint_vec_reps.shape)

    model_inputs = tf.split(joint_vec_reps,
                            num_or_size_splits=VEC_SIZES,
                            axis=1)

    out_reps = model.predict(model_inputs)

    return {k: out_reps[i, :] for i, k in enumerate(id_list)}
