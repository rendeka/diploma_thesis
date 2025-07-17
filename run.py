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
#             "--augment": [
#                 None,
#                 "cutmix",
#                 "mixup",
#                 ("tailored", "cutmix", None, "cutmix", None),
#                 ],            
#             "--decay": ["plateau"],
#             "--depth": [3],
#             # "--early_stopping": [True],
#             "--epochs": [150],
#             "--filters": [32],
#             # "--head": ["sigmoid"],
#             "--logdir_suffix": ["opt_aug"],
#             "--learning_rate": [0.01],
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


# # Convolution type
# runs.append(
#     RUN(
#         name="conv_type",
#         args_combinations={
#             "--activation": ["relu"],
#             # "--augment": [None],
#             "--conv_type": ["standard", "ds"],
#             "--decay": ["cosine"],
#             "--depth": [3],
#             "--early_stopping": [True],
#             "--epochs": [150],
#             "--filters": [32],
#             # "--head": ["sigmoid"],
#             "--logdir_suffix": ["conv_type"],
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

# # Depth
# runs.append(
#     RUN(
#         name="depth",
#         args_combinations={
#             "--activation": ["gelu"],
#             # "--augment": [None],
#             # "--conv_type": ["standard", "ds"],
#             "--decay": ["cosine"],
#             "--depth": [2, 3, 4],
#             "--early_stopping": [True],
#             "--epochs": [150],
#             "--filters": [32],
#             # "--head": ["sigmoid"],
#             "--logdir_suffix": ["depth"],
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
# # Pooling
# # runs.append(
# #     RUN(
# #         name="pooling",
# #         args_combinations={
# #             "--activation": ["relu"],
# #             # "--augment": [None],
# #             # "--conv_type": ["standard", "ds"],
# #             "--decay": ["cosine"],
# #             "--depth": [3],
# #             "--early_stopping": [True],
# #             "--epochs": [150],
# #             "--filters": [32],
# #             # "--head": ["sigmoid"],
# #             "--logdir_suffix": ["pooling"],
# #             "--learning_rate": [0.1],
# #             "--loss": ["KLD"],
# #             "--model": ["model5"],
# #             # "--optimizer": ["SGD", "Adam", "RMSprop"],
# #             "--pooling": ["max", "average", "no"],
# #             "--save_model": [True],
# #             "--scope": ["sub"],
# #             "--seed": [5, 6, 7, 8],
# #         },
# #         n_threads = 4    
# #     )
# # )

# # Stride
# # runs.append(
# #     RUN(
# #         name="stride",
# #         args_combinations={
# #             "--activation": ["relu"],
# #             # "--augment": [None],
# #             # "--conv_type": ["standard", "ds"],
# #             "--decay": ["cosine"],
# #             "--depth": [3],
# #             "--early_stopping": [True],
# #             "--epochs": [150],
# #             "--filters": [32],
# #             # "--head": ["sigmoid"],
# #             "--logdir_suffix": ["stride"],
# #             "--learning_rate": [0.1],
# #             "--loss": ["KLD"],
# #             "--model": ["model5"],
# #             # "--optimizer": ["SGD", "Adam", "RMSprop"],
# #             # "--pooling": ["max", "average", "no"],
# #             "--save_model": [True],
# #             "--stride": [1, 2],
# #             "--scope": ["sub"],
# #             "--seed": [5, 6, 7, 8],
# #         },
# #         n_threads = 4    
# #     )
# # )

# # FAG
# runs.append(
#     RUN(
#         name="FAG",
#         args_combinations={
#             "--activation": ["relu"],
#             # "--augment": [None],
#             # "--conv_type": ["standard", "ds"],
#             "--decay": ["plateau"],
#             "--depth": [3],
#             # "--early_stopping": [True],
#             "--epochs": [150],
#             "--fag": ["GAP", "Flatten"],
#             "--filters": [32],
#             # "--head": ["sigmoid"],
#             "--logdir_suffix": ["FAG5"],
#             "--learning_rate": [0.1],
#             "--loss": ["KLD"],
#             "--model": ["model5"],
#             # "--optimizer": ["SGD", "Adam", "RMSprop"],
#             # "--pooling": ["max", "average", "no"],
#             "--save_model": [True],
#             "--scope": ["sub"],
#             "--seed": [6, 7, 8],
#         },
#         n_threads = 3    
#     )
# )

# # FAG
# runs.append(
#     RUN(
#         name="activation",
#         args_combinations={
#             "--activation": ["relu", "selu", "linear", "leaky_relu", "gelu", "elu", "sigmoid"],
#             # "--augment": [None],
#             # "--conv_type": ["standard", "ds"],
#             "--decay": ["plateau"],
#             "--depth": [3],
#             # "--early_stopping": [True],
#             "--epochs": [150],
#             "--fag": ["GAP"],
#             "--filters": [32],
#             # "--head": ["sigmoid"],
#             "--logdir_suffix": ["activation"],
#             "--learning_rate": [0.1],
#             "--loss": ["KLD"],
#             "--model": ["model5"],
#             # "--optimizer": ["SGD", "Adam", "RMSprop"],
#             # "--pooling": ["max", "average", "no"],
#             "--save_model": [True],
#             "--scope": ["sub"],
#             "--seed": [6, 7, 8],
#         },
#         n_threads = 3    
#     )
# )

# Dropout
# runs.append(
#     RUN(
#         name="dropout",
#         args_combinations={
#             "--activation": ["relu"],
#             "--augment": [
#                 None,
#                 "cutmix",
#                 "mixup"
#                 ],           
#             "--decay": ["plateau"],
#             "--depth": [3],
#             "--dropout": [0, 0.3],
#             "--early_stopping": [True],
#             "--epochs": [450],
#             "--filters": [32],
#             # "--head": ["sigmoid"],
#             "--logdir_suffix": ["dropout3"],
#             "--learning_rate": [0.1],
#             "--loss": ["KLD"],
#             "--model": ["model5"],
#             "--optimizer": ["RMSprop"],
#             "--save_model": [True],
#             "--scope": ["sub"],
#             "--seed": [6, 7, 8],
#             "--spatial_dropout": [0, 0.2],
#         },
#         n_threads = 3    
#     )
# )

# New loss

# Hope for the best

# runs.append(
#     RUN(
#         name="Augmentation",
#         args_combinations={
#             "--activation": ["relu"],
#             "--augment": [
#                 None,
#                 "cutmix",
#                 "mixup",
#                 ("tailored", "cutmix", None, "cutmix", None),
#                 ],   
#             # "--batch_norm": ["False"],         
#             # "--conv_type": ["standard", "ds"],
#             "--decay": ["plateau"],
#             "--depth": [3],
#             # "--dropout": [0.1],
#             "--early_stopping": ["True"],
#             "--epochs": [200],
#             # "--fag": ["Flatten", "GAP"],
#             "--filters": [32],
#             # "--head": ["sigmoid"],
#             "--logdir_suffix": ["augment"],
#             "--learning_rate": [0.01],
#             "--loss": ["KLD"],
#             "--model": ["model5"],
#             # "--optimizer": ["SGD", "Adam", "RMSprop"],
#             # "--padding": ["valid", "same"],
#             "--save_model": ["True"],
#             "--scope": ["sub"],
#             "--seed": [6, 7, 8],
#             "--spatial_dropout": [0.1],
#         },
#         n_threads = 3    
#     )
# )

# runs.append(
#     RUN(
#         name="head",
#         args_combinations={
#             "--activation": ["relu"],
#             # "--augment": [
#             #     None,
#             #     "cutmix",
#             #     "mixup",
#             #     ("tailored", "cutmix", None, "cutmix", None),
#             #     ],   
#             # "--batch_norm": ["False"],         
#             # "--conv_type": ["standard", "ds"],
#             "--decay": ["plateau"],
#             "--depth": [3],
#             # "--dropout": [0.1],
#             "--early_stopping": ["True"],
#             "--epochs": [300],
#             # "--fag": ["Flatten", "GAP"],
#             "--filters": [32],
#             "--head": ["sigmoid", "softmax"],
#             "--logdir_suffix": ["head_local2"],
#             "--learning_rate": [0.1],
#             "--loss": ["KLD"],
#             "--model": ["model5"],
#             # "--optimizer": ["RMSprop"],
#             # "--padding": ["valid", "same"],
#             "--save_model": ["True"],
#             "--scope": ["sub"],
#             "--seed": [7],
#             # "--spatial_dropout": [0.1],
#         },
#         n_threads = 1    
#     )
# )

# runs.append(
#     RUN(
#         name="Augmentation + phase",
#         args_combinations={
#             "--activation": ["relu"],
#             "--augment": [
#                 # None,
#                 "cutmix",
#                 "mixup",
#                 # ("tailored", "cutmix", None, "cutmix", None),
#                 ],   
#             # "--batch_norm": ["False"],         
#             # "--conv_type": ["standard", "ds"],
#             "--decay": ["plateau"],
#             "--depth": [3, 4],
#             # "--dropout": [0.1],
#             "--early_stopping": ["True"],
#             "--epochs": [400],
#             # "--fag": ["Flatten", "GAP"],
#             "--filters": [32, 64],
#             # "--head": ["sigmoid"],
#             "--logdir_suffix": ["good_augment"],
#             "--learning_rate": [0.1],
#             "--loss": ["KLD"],
#             "--model": ["model5"],
#             "--optimizer": ["SGD", "RMSprop"],
#             # "--padding": ["valid", "same"],
#             # "--phase_augment": [("None", "cutmix", "cutmix")],
#             "--save_model": ["True"],
#             "--scope": ["sub"],
#             "--seed": [6, 7],
#             # "--spatial_dropout": [0.1],
#         },
#         n_threads = 3    
#     )
# )

# runs.append(
#     RUN(
#         name="ffn",
#         args_combinations = {
#             # "--activation": [
#             #     "celu", "elu", "exponential", "gelu", "glu", "hard_shrink", "hard_sigmoid", 
#             #     "hard_silu", "hard_swish", "hard_tanh", "leaky_relu", "linear", "log_sigmoid", 
#             #     "log_softmax", "mish", "relu", "relu6", "selu", "sigmoid", "silu", "swish", 
#             #     "soft_shrink", "softmax", "softplus", "softsign", "squareplus", "tanh", "tanh_shrink"
#             #     ],
#             "--activation": ["gelu"],
#             # "--alpha_dropout": [True],
#             "--augment": [
#                 None,
#                 "cutmix",
#                 "mixup",
#                 ("tailored", "cutmix", None, "cutmix", None),
#                 ],            
#             # "--batch_size": [16],
#             # "--bias_regularizer": [0],
#             # "--conv_type": ["ds"],
#             # "--dataloader_workers": [0],
#             "--decay": ["plateau"],
#             # "--depth": [1, 2, 3, 4, 5, 6],
#             "--dropout": [0.4],
#             # "---spatial_dropout": [0.0, 0.2],
#             "--epochs": [250],
#             # "--fag": ["GAP"],
#             "--filters": [64],
#             # "--ffm": [False],
#             # "--head": ["sigmoid"],
#             # "--kernel_regularizer": [0],
#             # "--kernel_size": [3],
#             # "--label_smoothing": 0.0,
#             "--learning_rate": [0.01],
#             # "--learning_rate_final": 0.001,
#             "--logdir_suffix": ["augment"],
#             "--loss": ["KLD"],
#             "--model": ["ffn"],
#             # "--optimizer": ["AdamW"],
#             # "--padding": ["same"],
#             # "--pooling": ["average"],
#             # "--seed": [42],
#             "--save_model": ["True"],
#             "--seed": [6, 7, 8],
#             # "--stochastic_depth": [0.0],
#             # "--stride": [1, 2],
#             # "--threads": [1],
#             "--weight_decay": [0.5],
#             # "--width": [1]
#         }
#     )
# )

# runs.append(
#     RUN(
#         name="cutmix",
#         args_combinations={
#             "--activation": ["relu"],
#             "--augment": [
#                 # None,
#                 "cutmix",
#                 "mixup",
#                 # ("tailored", "cutmix", None, "cutmix", None),
#                 ],   
#             # "--batch_norm": ["False"],         
#             # "--conv_type": ["standard", "ds"],
#             "--decay": ["plateau"],
#             "--depth": [3],
#             # "--dropout": [0.1],
#             "--early_stopping": ["True"],
#             "--epochs": [4],
#             # "--fag": ["Flatten", "GAP"],
#             "--filters": [32],
#             # "--head": ["sigmoid"],
#             "--logdir_suffix": ["augment_cm_try"],
#             "--learning_rate": [0.1],
#             "--loss": ["KLD"],
#             "--model": ["model5"],
#             "--optimizer": ["SGD"],
#             # "--padding": ["valid", "same"],
#             # "--phase_augment": [("None", "cutmix", "cutmix")],
#             "--save_model": ["True"],
#             "--scope": ["sub"],
#             "--seed": [6],
#             # "--spatial_dropout": [0, 0.1],
#         },
#         n_threads = 2
#     )
# )

runs.append(
    RUN(
        name="architectures",
        args_combinations={
            "--activation": ["relu"],
            # "--augment": [
            #     # None,
            #     "cutmix",
            #     "mixup",
            #     # ("tailored", "cutmix", None, "cutmix", None),
            #     ],   
            # "--batch_norm": ["False"],         
            # "--conv_type": ["standard", "ds"],
            "--decay": ["plateau"],
            "--depth": [1],
            # "--dropout": [0.1],
            "--early_stopping": ["True"],
            "--epochs": [2],
            # "--fag": ["Flatten", "GAP"],
            "--filters": [32],
            # "--head": ["sigmoid"],
            "--logdir_suffix": ["architectures"],
            "--learning_rate": [0.1],
            "--loss": ["KLD"],
            "--model": ["model5", "resnet", "ffn", "cbam", "se"],
            "--optimizer": ["SGD"],
            # "--padding": ["valid", "same"],
            # "--phase_augment": [("None", "cutmix", "cutmix")],
            "--save_model": ["True"],
            "--scope": ["sub"],
            "--seed": [7, 8],
            # "--spatial_dropout": [0, 0.1],
        },
        n_threads = 1
    )
)

if __name__ == "__main__":
    for run in runs:
        run.run()