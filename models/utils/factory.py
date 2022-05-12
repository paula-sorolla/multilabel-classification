import logging
import timm # What is trimn?????

logger = logging.getLogger(__name__)

from ..tresnet import TResnetM, TResnetL, TResnetXL


def create_model(args):
    """
        Create a model
    """

    model = Classifier(n_train_steps, n_labels, do_prob, bert_model=bert_model)

    return model
