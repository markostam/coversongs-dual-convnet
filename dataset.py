
from __future__ import absolute_import, division, print_function

import tensorflow as tf

import numpy as np
import itertools
import librosa
import os
import multiprocess
from io import open
from tqdm import tqdm

class SiameseDataset(object):

    def __init__(self,
                 train_txt_path=None,
                 test_txt_path=None,
                 train_mp3_path=None,
                 test_mp3_path=None,
                 sr=22050,
                 num_audio_channels=2,
                 spec_window_len_sec = 10,
                 spec_window_hop_sec = 2,
                 predict_or_eval_mode=None,
                 buff_size=15000,
                 ):

        self.predict_or_eval_mode = predict_or_eval_mode
        self.buff_size = buff_size

        self.sr = sr

        self.spec_window_len_samples = 441 # about 10 sec with 512 hop cqt #spec_window_len_sec * sr
        self.spec_hop_len_samples = 88 # about 2 sec with 512 hop cqt #spec_window_hop_sec * sr

        self.train_txt_path = train_txt_path
        self.test_txt_path = test_txt_path
        self.train_mp3_path = train_mp3_path
        self.test_mp3_path = test_mp3_path
        self.num_audio_channels = num_audio_channels

        print("Reading train cliques.")
        self.train_cliques = self._txt_to_cliques(train_txt_path, self.train_mp3_path)
        print("Reading test cliques.")
        # self.test_cliques = self._txt_to_cliques(test_txt_path, self.test_mp3_path)
        print("Getting positive train pairs.")
        self.pos_train_pairs = list(self._get_pos_pairs(self.train_cliques))
        print("Getting negative train pairs.")
        # ds = tf.data.Dataset.from_generator(self._get_neg_pairs_gen, (tf.string, tf.string),
        #                                     (tf.TensorShape([]),tf.TensorShape([])))
        # sess = tf.InteractiveSession()
        # ds2 = ds.make_one_shot_iterator()
        # print(sess.run(ds2.get_next()))
        # self.neg_train_pairs = self._get_neg_pairs_gen(self.train_cliques)

        # self.neg_train_pairs = list(self._get_neg_pairs_multi(self.train_cliques))
        self.pos_repeat_count = 3000

        print("Getting positive test pairs.")
        # self.pos_test_pairs = set(self._get_pos_pairs(self.test_cliques))
        print("Getting negative test pairs.")
        # self.neg_test_pairs = set(self._get_duped_neg_pairs(self.test_cliques))

        self.dataset = self._make_dataset()

    def _make_dataset(self):
        """
        Create the actual dataset.

            Returns:
                tf.data.Dataset: The dataset that generates input to the network.
        """


        if self.predict_or_eval_mode is not None:
            pos_filepaths = tf.data.Dataset.from_tensor_slices(list(self.pos_test_pairs))
            # neg_filepaths = tf.data.Dataset.from_tensor_slices(list(self.neg_test_pairs))
        else:
            pos_filepaths = tf.data.Dataset.from_tensor_slices(list(self.pos_train_pairs))
            neg_filepaths = tf.data.Dataset.from_generator(self._get_neg_pairs_gen, (tf.string, tf.string),
                                           (tf.TensorShape([]), tf.TensorShape([])))
            # neg_filepaths = tf.data.Dataset.from_tensor_slices(list(self.neg_train_pairs))

        pos_filepaths = pos_filepaths.map(lambda x: ((x[0],x[1]), tf.constant(1)))
        neg_filepaths = neg_filepaths.map(lambda x, y: ((x, y), tf.constant(0)))

        if self.predict_or_eval_mode is None:
            pos_filepaths = pos_filepaths.apply(
                tf.contrib.data.shuffle_and_repeat(buffer_size=self.buff_size, count=self.pos_repeat_count))

        # zip repeated positive and negative together for training
        dataset = tf.data.Dataset.zip((pos_filepaths, neg_filepaths)).flat_map(
                lambda files, label: tf.data.Dataset.from_tensors(files).concatenate(
                    tf.data.Dataset.from_tensors(label)))

        dataset.prefetch(1)

        dataset = dataset.map(self._read_audio_files, num_parallel_calls=4)
        dataset = dataset.map(self._cqt, num_parallel_calls=4)
        dataset = dataset.map(self._window_signals)
        dataset = dataset.map(self._cartesian_sliced_tensors_and_tiled_labels)
        import pdb; pdb.set_trace()
        dataset = dataset.apply(tf.contrib.data.unbatch())

        sess = tf.InteractiveSession()
        ds = dataset.make_one_shot_iterator().get_next()
        ds2 = sess.run(ds)
        print(ds2[0]['song1']['windowed_cqt'].shape)
        print(ds2[0]['song2']['windowed_cqt'].shape)
        print(ds2[1]['label_int'].shape)

        print(ds2)

    def _read_audio_files(self, filepaths, labels):
            """
             Read the audio file and return a 1-d tensor of the waveform

                Args:
                    filenames tuple(str): Tuple of filename pair of audio files.
                    filepath str: Path to the filename pair.

                Returns:
                    dict(song1: dict(tensor(waveform, float32)), filepath: str)):
                        Nested dict of waveform and filepath for song1.
                    dict(song2: dict(tensor(waveform, float32)), filepath: str)):
                        Nested dict of waveform and filepath for song2.
            """

            waveforms = []
            for filepath in filepaths:
                audio_binary = tf.read_file(filepath)
                waveform = tf.contrib.ffmpeg.decode_audio(
                    audio_binary,
                    file_format='mp3',
                    samples_per_second=self.sr,
                    channel_count=self.num_audio_channels)
                if self.num_audio_channels > 1:
                    waveform = tf.reduce_mean(waveform, axis=1, name="waveform_to_mono")
                waveforms.append(waveform)

            return ({"song1": {"waveform": waveforms[0], "filepath": filepaths[0]},
                     "song2": {"waveform": waveforms[1], "filepath": filepaths[1]}},
                    {"label_int": labels})

    def _cqt(self, songs, labels):
        songs_ret = {}
        for key in songs:
            waveform = songs[key]["waveform"]
            filepath = songs[key]["filepath"]
            songs_ret[key] = {"cqt": tf.py_func(self._cqt_pyfunc, [waveform], tf.float32), "filepath": filepath}
        return songs_ret, labels

    def _cqt_pyfunc(self, waveform):
        C = librosa.cqt(y=waveform, sr=self.sr, hop_length=512)
        C = np.log(np.abs(C)).astype(np.float32)
        return C

    def _window_signals(self, songs, labels):
        """
          Apply a window function on the waveform to split it into equal pieces

             Args:
                 dict(waveform: tensor(waveform, float32)):
                     A dictionary of tensors of the audio waveform of type float32.
                 dict(filepath': tensor(filepath, str)):
                     filepath: the original filepath of this spectrogram window (str).

             Returns:
                 dict(windowed_waveform: tensor(windowed_waveform, float32)):
                     windowed_waveform: tensors of the windowed audio waveform of type float32.
                 dict(filepath': tensor(filepath, str)):
                     filepath: the original filepath of this spectrogram window (str).
         """
        songs_ret = {}
        for key in songs:
            cqt = songs[key]["cqt"]
            filepath = songs[key]["filepath"]
            # Split data into windows of duration window_length_sec
            windowed_cqt = tf.contrib.signal.frame(
                cqt, self.spec_window_len_samples, self.spec_hop_len_samples)
            songs_ret[key] = {"windowed_cqt": windowed_cqt, "filepath": filepath}

        return songs_ret, labels

    def _cartesian_sliced_tensors_and_tiled_labels(self, songs, labels):
        """
        Take the two windowed CQT's and combine them in pairs of two with a cartesian product along the
        first axis. Returns a [N, 2, CQT_rows, CQT_cols] tensor.
        :param songs:
        :return:
        """

        song1 = songs["song1"]["windowed_cqt"][None, :, None]
        song2 = songs["song1"]["windowed_cqt"][:, None, None]
        cartesian_product = tf.concat([song1 + tf.zeros_like(song2),
                                       tf.zeros_like(song1) + song2], axis=2)[0]
        # shift to [2, N_Tiled, CQT_rows, CQT_cols]
        cartesian_product = tf.transpose(cartesian_product, perm=[1,2,0,3], name="cartesian_transposed")
        song1 = cartesian_product[0]
        song2 = cartesian_product[1]

        # tile the darn labels
        padded_labels = {}
        tile_dim = tf.shape(cartesian_product)[1]
        for key in labels:
            label = labels[key]
            label = tf.tile(tf.expand_dims(label,0), [tile_dim])
            padded_labels[key] = label

        padded_filepaths = {}
        for key in songs:
            filepath = songs[key]['filepath']
            filepath = tf.tile(tf.expand_dims(filepath, 0), [tile_dim])
            padded_filepaths[key] = filepath

        return ({"song1": {"windowed_cqt": song1, "filepath": padded_filepaths['song1']},
                 "song2": {"windowed_cqt": song2, "filepath": padded_filepaths['song2']}},
                padded_labels)
        # this is faster maybe but less clear
        # return ({"cqt_two_songs": cartesian_product,
        #         "filepath_1": songs["song1"]["filepath"],
        #         "filepath2": songs["song2"]["filepath"]}, padded_labels)


    def _txt_to_cliques(self, txt_path, mp3_folder_path):
        '''
        reads a textfile of song cliques in shs
        creates a dictionary out of second hand songset 'cliques'
        or groups of cover songs and returns the dict of cliques
        based on their msd id
        '''
        with open(txt_path, encoding='utf-8') as shs_src:
            shs = list(shs_src)
            shs = shs[14:]
        cliques = {}
        for ent in shs:
            ent = ent.replace('\n', '')
            if ent[0] == '%':
                tempKey = ent.lower()
                cliques[tempKey] = []
            else:
                filename = ent.split("<SEP>")[0] + '.mp3'
                filepath = os.path.join(mp3_folder_path, filename)
                cliques[tempKey].append(filepath)
        return cliques

    def _get_pos_pairs(self, cliques):
        """
        Function to generate all possible positive examples given a dictionary of cliques

        Args:
            cliques: dict(str: list(str)): dict of clique name to all audio files in said clique

        Returns:
            list(tuple(str,str)): all possible positive combinations from cliques
        """
        positive_examples = (list(itertools.combinations(val, 2)) for key, val in cliques.items())
        positive_examples = [i for j in positive_examples for i in j]
        return positive_examples

    # def _get_neg_pairs(self, cliques):
    #     """
    #     Function to generate all possible negative examples given a dictionary of cliques
    #
    #     Args:
    #         dict(str: list(str)): dict of clique name to all audio files in said clique
    #
    #     Yields:
    #         set(str,str): all negative combinations from cliques
    #     """
    #     # running_list = []
    #     neg_combo_set = set()
    #     for clique_name, clique in tqdm(cliques.items(), total=len(cliques)):
    #         for song in clique:
    #             for clique_name_2, clique_2 in cliques.items():
    #                 if clique_name_2 != clique_name:
    #                     for song_2 in clique_2:
    #                         neg_combo = tuple(sorted((song, song_2)))
    #                         neg_combo_set.add(neg_combo)
    #     return neg_combo_set
    #
    # def _get_neg_pairs_multi(self, cliques, poolsize=8):
    #     """
    #     Function to generate all possible negative examples given a dictionary of cliques
    #
    #     Args:
    #         dict(str: list(str)): dict of clique name to all audio files in said clique
    #
    #     Yields:
    #         set(str,str): all negative combinations from cliques
    #     """
    #     neg_combo_set = set()
    #     p = multiprocess.Pool(poolsize)
    #
    #     def _imap_func(name_clique_tup):
    #         neg_combos = set()
    #         clique_name, clique = name_clique_tup
    #         for song in clique:
    #             for clique_name_2, clique_2 in cliques.items():
    #                 if clique_name_2 != clique_name:
    #                     for song_2 in clique_2:
    #                         neg_combo = tuple(sorted((song, song_2)))
    #                         neg_combos.add(neg_combo)
    #         return neg_combos
    #
    #     imap_arg = ((clique_name, clique) for clique_name, clique in cliques.items())
    #
    #     for neg_combos in tqdm(p.imap(_imap_func, imap_arg), total=len(cliques)):
    #         neg_combo_set.update(neg_combos)
    #
    #     return neg_combo_set

    def _get_neg_pairs_gen(self):
        """
        Function to generate all possible negative examples given a dictionary of cliques

        Args:
            dict(str: list(str)): dict of clique name to all audio files in said clique

        Yields:
            tuple(str,str): all negative combinations from cliques
        """
        running_list = []
        for clique_name, clique in self.train_cliques.items():
            for song in clique:
                for clique_name_2, clique_2 in self.train_cliques.items():
                    if clique_name_2 != clique_name:
                        for song_2 in clique_2:
                            neg_combo = tuple(sorted((song, song_2)))
                            if neg_combo not in running_list:
                                running_list.append(neg_combo)
                                yield neg_combo

siamese = SiameseDataset(
                 train_txt_path="/Users/markostamenovic/code/shs/shs_train/shs_dataset_train.txt",
                 train_mp3_path="/Users/markostamenovic/code/shs/shs_train/",
                 sr=22050,
                 num_audio_channels=2,
                 spec_window_len_sec=10,
                 spec_window_hop_sec=2,
                 )