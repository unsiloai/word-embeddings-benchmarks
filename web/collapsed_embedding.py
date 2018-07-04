import logging
import copy
import numpy as np
from collections import defaultdict

from web.embedding import Embedding

logger = logging.getLogger(__name__)


class CollapsedEmbedding(Embedding):
    def __init__(self, vocabulary, vectors, collapsing_mappings):
        super(CollapsedEmbedding, self).__init__(vocabulary, vectors)

        self.collapsing_mappings = collapsing_mappings
        self.inverse_collapsing_map = invert_dict_with_lists(self.collapsing_mappings)

        # Add collapsed words to vocabulary but save copy of of original vocabulary
        self.embedding_vocabulary = copy.deepcopy(self.vocabulary)
        self.add_collapsed_to_vocabulary(self.collapsing_mappings)

    def get_collapsed_key(self, k):
        return self.inverse_collapsing_map[k]

    def __getitem__(self, k):
        try:
            return self.vectors[self.embedding_vocabulary[k]]
        except KeyError:
            try:
                return self.vectors[self.vocabulary[self.get_collapsed_key(k)]]
            except KeyError:
                return None

    def __delitem__(self, k):
        """Remove the word and its vector from the embedding.

        Note:
         This operation costs \\theta(n). Be careful putting it in a loop.
        """
        index = self.vocabulary[k]
        del self.vocabulary[k]
        try:
            self.vectors = np.delete(self.vectors, index, 0)
        except:
            pass

    def add_collapsed_to_vocabulary(self, dict_):
        """
        Adds values in dict_ to vocabulary if the key is in the vocabulary.
        """
        for k, vals in dict_.items():
            if k in self.vocabulary:
                for v in vals:
                    self.vocabulary.add(v)


def invert_dict_with_lists(dict_):
    """
    Convert dictionary where value is a list to
    {val1:key1, val2:key1, val3:key1}.

    :param dict_: dictionary
    :return: defaultdict with one entrance per value in input.
    """
    new_dict = defaultdict(str)
    for k, vals in dict_.items():
        for v in vals:
            new_dict[v] = k
    return new_dict
