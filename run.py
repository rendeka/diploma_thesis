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

# runs.append(
#     RUN(
#         name="ffn",
#         args_combinations={
#             "--activation": ["sigmoid"],
#             # "--augment": [None, "cutmix", "mixup", ("cutmix", "mixup")],
#             "--decay": ["cosine"],
#             "--epochs": [100],
#             "--filters": [32, 64, 128],
#             "--head": ["sigmoid"],
#             "--logdir_suffix": ["base"],
#             "--learning_rate": [0.01],
#             "--loss": ["MSE"],
#             "--model": ["ffn"],
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

runs.append(
    RUN(
        name="decay",
        args_combinations = {
            "--activation": ["relu"],
            # "--alpha_dropout": [True],
            "--augment": [
                ("tailored", "cutmix", None, "cutmix", None),
                ],
            "--batch_size": [32],
            "--decay": ["plateau"],
            "--depth": [3],
            # "--dropout": [0.0, 0.2],
            # "---spatial_dropout": [0.0, 0.2],
            "--epochs": [400],
            "--fag": ["GAP"],
            "--filters": [32],
            "--ffm": [True],
            "--head": ["sigmoid", "softmax"],
            # "--learning_rate": [0.001],
            "--logdir_suffix": ["head-loss"],
            "--loss": ["KLD", "CCE"],
            "--model": ["model5"],
            # "--padding": ["periodic", "same"],
            "--phase_augment": [
                # (None, None, None),
                (None, None, "cutmix"),
                ],
            "--seed": [5, 6, 7],
            "--save_model": [True],
            "--scope": ["sub"],
            # "--trans_probs": [True],
        },
        n_threads = 4
    )
)

# runs.append(
#     RUN(
#         name="head",
#         args_combinations = {
#             "--activation": ["relu"],
#             # "--alpha_dropout": [True],
#             "--augment": [
#                 ("tailored", "cutmix", None, "cutmix", None),
#                 ],
#             "--decay": ["cosine"],
#             "--depth": [3],
#             # "--dropout": [0.0, 0.2],
#             # "---spatial_dropout": [0.0, 0.2],
#             "--epochs": [200],
#             "--fag": ["GAP"],
#             "--filters": [32],
#             "--ffm": [True],
#             "--head": ["sigmoid"],
#             # "--learning_rate": [0.001],
#             "--logdir_suffix": ["head"],
#             "--loss": ["KLD"],
#             "--model": ["model5"],
#             "--phase_augment": [
#                 # (None, None, None),
#                 (None, None, "cutmix"),
#                 ],
#             "--seed": [1, 2],
#             "--save_model": [True],
#             "--scope": ["sub"],
#             # "--trans_probs": [True],
#         },
#         n_threads = 2   
#     )
# )

# runs.append(
#     RUN(
#         name="head",
#         args_combinations = {
#             "--activation": ["relu"],
#             # "--alpha_dropout": [True],
#             "--augment": [
#                 ("tailored", "cutmix", None, "cutmix", None),
#                 ],
#             "--decay": ["cosine"],
#             "--depth": [3],
#             # "--dropout": [0.0, 0.2],
#             # "---spatial_dropout": [0.0, 0.2],
#             "--epochs": [200],
#             "--fag": ["GAP"],
#             "--filters": [32],
#             "--ffm": [True],
#             "--head": ["softmax"],
#             # "--learning_rate": [0.001],
#             "--logdir_suffix": ["head"],
#             "--loss": ["CCE"],
#             "--model": ["model5"],
#             "--phase_augment": [
#                 # (None, None, None),
#                 (None, None, "cutmix"),
#                 ],
#             "--seed": [1, 2],
#             "--save_model": [True],
#             "--scope": ["sub"],
#             # "--trans_probs": [True],
#         },
#         n_threads = 2
#     )
# )




# runs.append(
#     RUN(
#         name="new_augment",
#         args_combinations = {
#             "--activation": ["relu"],
#             # "--alpha_dropout": [True],
#             "--augment": [
#                 None,
#                 "cutmix",
#                 "mixup",
#                 ("tailored", "cutmix", None, None, "mixup"),
#                 ],
#             "--decay": ["cosine"],
#             # "--dropout": [0.0, 0.2],
#             # "---spatial_dropout": [0.0, 0.2],
#             "--epochs": [150],
#             "--filters": [64],
#             "--head": ["sigmoid"],
#             "--learning_rate": [0.01],
#             "--logdir_suffix": ["new_augment"],
#             "--loss": ["KLD", "MSE"],
#             "--model": ["ffn"],
#             "--save_model": [True],
#             "--scope": ["sub"],
#             "--trans_probs": [True],
#         },
#         n_threads = 4
#     )
# )

if __name__ == "__main__":
    for run in runs:
        run.run()