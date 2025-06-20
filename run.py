#!/usr/bin/env python3
from running import RUN

runs = []
# runs.append(
#     RUN(
#         name="softmax",
#         args_combinations = {
#             # "--activation": [
#             #     "celu", "elu", "exponential", "gelu", "glu", "hard_shrink", "hard_sigmoid", 
#             #     "hard_silu", "hard_swish", "hard_tanh", "leaky_relu", "linear", "log_sigmoid", 
#             #     "log_softmax", "mish", "relu", "relu6", "selu", "sigmoid", "silu", "swish", 
#             #     "soft_shrink", "softmax", "softplus", "softsign", "squareplus", "tanh", "tanh_shrink"
#             #     ],
#             "--activation": ["relu"],
#             # "--alpha_dropout": [True],
#             # "--augment": [None, "cutmix", "mixup", ("cutmix", "mixup")],
#             # "--batch_size": [16],
#             # "--bias_regularizer": [0],
#             # "--conv_type": ["ds"],
#             # "--dataloader_workers": [0],
#             "--decay": ["cosine"],
#             "--depth": [1, 2, 3, 4, 5, 6],
#             # "--dropout": [0.0, 0.2],
#             # "---spatial_dropout": [0.0, 0.2],
#             "--epochs": [30],
#             "--fag": ["GAP"],
#             "--filters": [16, 32, 64],
#             # "--ffm": [False],
#             # "--head": ["sigmoid"],
#             # "--kernel_regularizer": [0],
#             # "--kernel_size": [3],
#             # "--label_smoothing": 0.0,
#             # "--learning_rate": [0.001],
#             # "--learning_rate_final": 0.001,
#             "--logdir_suffix": ["softmax-GAP"],
#             "--loss": ["CCE"],
#             "--model": ["model5"],
#             # "--optimizer": ["AdamW"],
#             # "--padding": ["same"],
#             # "--pooling": ["average"],
#             # "--seed": [42],
#             # "--save_model": False,
#             # "--stochastic_depth": [0.0],
#             # "--stride": [1, 2],
#             # "--threads": [1],
#             # "--weight_decay": [1e-5],
#             # "--width": [1]
#         }
#     )
# )

# runs.append(
#     RUN(
#         name="sigmoid",
#         args_combinations={
#             "--activation": ["relu", "gelu"],
#             "--augment": [None, "cutmix", "mixup", ("cutmix", "mixup")],
#             "--decay": ["cosine"],
#             "--depth": [4],
#             "--epochs": [80],
#             "--fag": ["GAP"],
#             "--filters": [32],
#             "--head": ["sigmoid"],
#             "--logdir_suffix": ["sigmoid-long"],
#             "--loss": ["KLD"],
#             "--model": ["model5"],
#         }
#     )
# )

# runs.append(
#     RUN(
#         name="softmax",
#         args_combinations={
#             "--activation": ["relu", "gelu"],
#             "--augment": [None, "cutmix", "mixup", ("cutmix", "mixup")],
#             "--decay": ["cosine"],
#             "--depth": [4],
#             "--epochs": [80],
#             "--fag": ["GAP"],
#             "--filters": [32],
#             "--head": ["softmax"],
#             "--logdir_suffix": ["softmax-long"],
#             "--loss": ["KLD"],
#             "--model": ["model5"],
#         }
#     )
# )

# runs.append(
#     RUN(
#         name="cbam",
#         args_combinations={
#             "--activation": ["relu", "gelu"],
#             "--augment": [None, "cutmix", "mixup", ("cutmix", "mixup")],
#             "--decay": ["cosine"],
#             "--depth": [1],
#             "--epochs": [80],
#             "--fag": ["GAP"],
#             "--filters": [32],
#             "--head": ["sigmoid"],
#             "--logdir_suffix": ["sigmoid-long"],
#             "--loss": ["KLD"],
#             "--model": ["cbam"],
#         }
#     )
# )


############## SUB-data runs

# runs.append(
#     RUN(
#         name="padding",
#         args_combinations = {
#             "--activation": ["relu"],
#             # "--alpha_dropout": [True],
#             "--augment": [
#                 None,
#                 "cutmix",
#                 ("tailored", "cutmix", None, "cutmix", None),
#                 ],
#             "--decay": ["cosine"],
#             "--depth": [3],
#             # "--dropout": [0.0, 0.2],
#             # "---spatial_dropout": [0.0, 0.2],
#             "--epochs": [1],
#             "--fag": ["GAP"],
#             "--filters": [32],
#             "--ffm": [True],
#             "--head": ["sigmoid"],
#             # "--learning_rate": [0.001],
#             "--logdir_suffix": ["padding"],
#             "--loss": ["KLD"],
#             "--model": ["model5"],
#             "--padding": ["periodic", "same", "valid"],
#             "--phase_augment": [
#                 # (None, None, None),
#                 (None, "cutmix", "cutmix"),
#                 ],
#             "--seed": [1, 2, 3],
#             "--save_model": [True],
#             "--scope": ["sub"],
#             # "--trans_probs": [True],
#         },
#         n_threads = 3
#     )
# )

## --------------FINAL RUNS--------------

# runs.append(
#     RUN(
#         name="loss",
#         args_combinations={
#             "--activation": ["relu"],
#             # "--augment": [None],
#             # "--decay": ["cosine"],
#             "--depth": [3],
#             "--early_stopping": [True],
#             "--epochs": [150],
#             # "--filters": [32, 64, 128],
#             "--filters": [32],
#             "--head": ["sigmoid"],
#             "--logdir_suffix": ["loss"],
#             "--learning_rate": [0.1],
#             "--loss": ["KLD", "MSE"],
#             "--model": ["model5"],
#             # "--optimizer": ["SGD"],
#             "--save_model": [True],
#             "--scope": ["sub"],
#             "--seed": [5, 6, 7, 8],
#         },
#         n_threads = 4      
#     )
# )

# runs.append(
#     RUN(
#         name="filters",
#         args_combinations={
#             "--activation": ["relu"],
#             # "--augment": [None],
#             # "--decay": ["cosine"],
#             "--depth": [3],
#             "--early_stopping": [True],
#             "--epochs": [150],
#             "--filters": [32, 64, 128],
#             "--head": ["sigmoid"],
#             "--logdir_suffix": ["filters"],
#             "--learning_rate": [0.1],
#             "--loss": ["KLD"],
#             "--model": ["model5"],
#             # "--optimizer": ["SGD"],
#             "--save_model": [True],
#             "--scope": ["sub"],
#             "--seed": [5, 6, 7, 8],
#         },
#         n_threads = 4     
#     )
# )

# runs.append(
#     RUN(
#         name="optimizer",
#         args_combinations={
#             "--activation": ["relu"],
#             # "--augment": [None],
#             # "--decay": ["cosine"],
#             "--depth": [3],
#             "--early_stopping": [True],
#             "--epochs": [150],
#             "--filters": [32],
#             "--head": ["sigmoid"],
#             "--logdir_suffix": ["optimizer"],
#             "--learning_rate": [0.1],
#             "--loss": ["KLD"],
#             "--model": ["model5"],
#             "--optimizer": ["SGD", "Adam", "RMSprop"],
#             "--save_model": [True],
#             "--scope": ["sub"],
#             "--seed": [5, 6, 7, 8],
#         },
#         n_threads = 4      
#     )
# )

# runs.append(
#     RUN(
#         name="activation",
#         args_combinations={
#             "--activation": ["linear", "relu", "gelu", "swish", "tanh"],
#             # "--augment": [None],
#             # "--decay": ["cosine"],
#             "--depth": [3],
#             "--early_stopping": [True],
#             "--epochs": [150],
#             "--filters": [32],
#             "--head": ["sigmoid"],
#             "--logdir_suffix": ["activation"],
#             "--learning_rate": [0.1],
#             "--loss": ["KLD"],
#             "--model": ["model5"],
#             # "--optimizer": ["SGD", "Adam", "RMSprop"],
#             "--save_model": [True],
#             "--scope": ["sub"],
#             "--seed": [5, 6, 7, 8],
#         },
#         n_threads = 4      
#     )
# )

# runs.append(
#     RUN(
#         name="normalization",
#         args_combinations={
#             "--activation": ["relu"],
#             # "--augment": [None],
#             # "--decay": ["cosine"],
#             "--depth": [3],
#             "--early_stopping": [True],
#             "--epochs": [150],
#             "--filters": [32],
#             "--head": ["sigmoid"],
#             "--logdir_suffix": ["normalization"],
#             "--learning_rate": [0.1],
#             "--loss": ["KLD"],
#             "--model": ["model5"],
#             # "--optimizer": ["SGD", "Adam", "RMSprop"],
#             "--save_model": [True],
#             "--scope": ["sub"],
#             "--seed": [5, 6, 7, 8],
#         },
#         n_threads = 4    
#     )
# )

# runs.append(
#     RUN(
#         name="head",
#         args_combinations={
#             "--activation": ["relu"],
#             # "--augment": [None],
#             # "--decay": ["cosine"],
#             "--depth": [3],
#             "--early_stopping": [True],
#             "--epochs": [150],
#             "--filters": [32],
#             "--head": ["sigmoid", "softmax"],
#             "--logdir_suffix": ["head"],
#             "--learning_rate": [0.1],
#             "--loss": ["KLD"],
#             "--model": ["model5"],
#             # "--optimizer": ["SGD", "Adam", "RMSprop"],
#             "--save_model": [True],
#             "--scope": ["sub"],
#             "--seed": [5, 6, 7, 8],
#         },
#         n_threads = 4    
#     )
# )

# runs.append(
#     RUN(
#         name="decay",
#         args_combinations={
#             "--activation": ["relu"],
#             # "--augment": [None],
#             "--decay": [None, "linear", "cosine", "exponential", "plateau"],
#             "--depth": [3],
#             "--early_stopping": [True],
#             "--epochs": [150],
#             "--filters": [32],
#             "--head": ["sigmoid"],
#             "--logdir_suffix": ["decay"],
#             "--learning_rate": [0.1],
#             "--loss": ["KLD"],
#             "--model": ["model5"],
#             # "--optimizer": ["SGD", "Adam", "RMSprop"],
#             "--save_model": [True],
#             "--scope": ["sub"],
#             "--seed": [5, 6, 7, 8],
#         },
#         n_threads = 4    
#     )
# )


# Convolution type
runs.append(
    RUN(
        name="conv_type",
        args_combinations={
            "--activation": ["relu"],
            # "--augment": [None],
            "--conv_type": ["standard", "ds"],
            "--decay": ["cosine"],
            "--depth": [3],
            "--early_stopping": [True],
            "--epochs": [150],
            "--filters": [32],
            # "--head": ["sigmoid"],
            "--logdir_suffix": ["conv_type"],
            "--learning_rate": [0.1],
            "--loss": ["KLD"],
            "--model": ["model5"],
            # "--optimizer": ["SGD", "Adam", "RMSprop"],
            "--save_model": [True],
            "--scope": ["sub"],
            "--seed": [5, 6, 7, 8],
        },
        n_threads = 4    
    )
)

# Depth
runs.append(
    RUN(
        name="depth",
        args_combinations={
            "--activation": ["relu"],
            # "--augment": [None],
            # "--conv_type": ["standard", "ds"],
            "--decay": ["cosine"],
            "--depth": [2, 3, 4],
            "--early_stopping": [True],
            "--epochs": [150],
            "--filters": [32],
            # "--head": ["sigmoid"],
            "--logdir_suffix": ["depth"],
            "--learning_rate": [0.1],
            "--loss": ["KLD"],
            "--model": ["model5"],
            # "--optimizer": ["SGD", "Adam", "RMSprop"],
            "--save_model": [True],
            "--scope": ["sub"],
            "--seed": [5, 6, 7, 8],
        },
        n_threads = 4    
    )
)
# Pooling
# runs.append(
#     RUN(
#         name="pooling",
#         args_combinations={
#             "--activation": ["relu"],
#             # "--augment": [None],
#             # "--conv_type": ["standard", "ds"],
#             "--decay": ["cosine"],
#             "--depth": [3],
#             "--early_stopping": [True],
#             "--epochs": [150],
#             "--filters": [32],
#             # "--head": ["sigmoid"],
#             "--logdir_suffix": ["pooling"],
#             "--learning_rate": [0.1],
#             "--loss": ["KLD"],
#             "--model": ["model5"],
#             # "--optimizer": ["SGD", "Adam", "RMSprop"],
#             "--pooling": ["max", "average", "no"],
#             "--save_model": [True],
#             "--scope": ["sub"],
#             "--seed": [5, 6, 7, 8],
#         },
#         n_threads = 4    
#     )
# )

# Stride
# runs.append(
#     RUN(
#         name="stride",
#         args_combinations={
#             "--activation": ["relu"],
#             # "--augment": [None],
#             # "--conv_type": ["standard", "ds"],
#             "--decay": ["cosine"],
#             "--depth": [3],
#             "--early_stopping": [True],
#             "--epochs": [150],
#             "--filters": [32],
#             # "--head": ["sigmoid"],
#             "--logdir_suffix": ["stride"],
#             "--learning_rate": [0.1],
#             "--loss": ["KLD"],
#             "--model": ["model5"],
#             # "--optimizer": ["SGD", "Adam", "RMSprop"],
#             # "--pooling": ["max", "average", "no"],
#             "--save_model": [True],
#             "--stride": [1, 2],
#             "--scope": ["sub"],
#             "--seed": [5, 6, 7, 8],
#         },
#         n_threads = 4    
#     )
# )

# FAG
runs.append(
    RUN(
        name="FAG",
        args_combinations={
            "--activation": ["relu"],
            # "--augment": [None],
            # "--conv_type": ["standard", "ds"],
            "--decay": ["cosine"],
            "--depth": [3],
            "--early_stopping": [True],
            "--epochs": [150],
            "--fag": ["GAP", "Flatten", "SE"],
            "--filters": [32],
            # "--head": ["sigmoid"],
            "--logdir_suffix": ["FAG"],
            "--learning_rate": [0.1],
            "--loss": ["KLD"],
            "--model": ["model5"],
            # "--optimizer": ["SGD", "Adam", "RMSprop"],
            # "--pooling": ["max", "average", "no"],
            "--save_model": [True],
            "--scope": ["sub"],
            "--seed": [5, 6, 7, 8],
        },
        n_threads = 4    
    )
)

# Dropout
runs.append(
    RUN(
        name="dropout",
        args_combinations={
            "--activation": ["relu"],
            # "--augment": [None],
            # "--conv_type": ["standard", "ds"],
            "--decay": ["cosine"],
            "--depth": [3],
            "--dropout": [0, 0.1, 0.2],
            "--early_stopping": [True],
            "--epochs": [150],
            "--filters": [32],
            # "--head": ["sigmoid"],
            "--logdir_suffix": ["dropout"],
            "--learning_rate": [0.1],
            "--loss": ["KLD"],
            "--model": ["model5"],
            # "--optimizer": ["SGD", "Adam", "RMSprop"],
            "--save_model": [True],
            "--scope": ["sub"],
            "--seed": [5, 6, 7, 8],
        },
        n_threads = 4    
    )
)

# New loss


if __name__ == "__main__":
    for run in runs:
        run.run()