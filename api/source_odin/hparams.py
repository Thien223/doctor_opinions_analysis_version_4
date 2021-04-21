
# backup
# class Hparams():
#     ################################
#     # Experiment Parameters        #
#     ################################
#     inception_inchannel= 8
#     bottneck_outchannel = 32
#     input_sequence_length=150
#     output_sequence_length=7
#     linear_in_dim=2400
#     num_classes=38
#     batch_size=32
class Hparams():
    ################################
    # Experiment Parameters        #
    ################################
    inception_inchannel= 8
    bottneck_outchannel = 32
    input_sequence_length=200
    output_sequence_length=15
    linear_in_dim=480
    num_classes=38
    batch_size=512
def create_hparams(hparams=None):
    """Create model hyperparameters. Parse nondefault from given string."""
    hparams = Hparams()
    return hparams

