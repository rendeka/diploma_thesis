#!/usr/bin/env python3
from running import RUN

runs = []

runs.append(
    RUN(
        name="praying a lot",
        args_combinations={
            "--activation": ["relu"],
            "--augment": [
                # None,
                "cutmix",
                "mixup",
                # ("tailored", "cutmix", None, "cutmix", None),
                ],   
            # "--batch_norm": ["False"],         
            # "--conv_type": ["standard", "ds"],
            "--decay": ["plateau"],
            "--depth": [3],
            # "--dropout": [0.1],
            "--early_stopping": ["True"],
            "--epochs": [400],
            # "--fag": ["Flatten", "GAP"],
            "--filters": [32],
            "--head": ["sigmoid", "softmax"],
            "--logdir_suffix": ["aug-god-please"],
            "--learning_rate": [0.1],
            "--loss": ["KLD"],
            "--model": ["model5"],
            "--optimizer": ["SGD"],
            # "--padding": ["valid", "same"],
            # "--phase_augment": [("None", "cutmix", "cutmix")],
            "--save_model": ["True"],
            "--scope": ["sub"],
            "--seed": [6, 7],
            # "--spatial_dropout": [0, 0.1],
        },
        n_threads = 2  
    )
)

# runs.append(
#     RUN(
#         name="mixup",
#         args_combinations={
#             "--activation": ["relu"],
#             "--augment": [
#                 # None,
#                 # "cutmix",
#                 "mixup",
#                 # ("tailored", "cutmix", None, "cutmix", None),
#                 ],   
#             # "--batch_norm": ["False"],         
#             # "--conv_type": ["standard", "ds"],
#             "--decay": ["plateau"],
#             "--depth": [3],
#             # "--dropout": [0.1],
#             "--early_stopping": ["True"],
#             "--epochs": [400],
#             # "--fag": ["Flatten", "GAP"],
#             "--filters": [32],
#             # "--head": ["sigmoid"],
#             "--logdir_suffix": ["mixup"],
#             "--learning_rate": [0.1],
#             "--loss": ["KLD"],
#             "--model": ["model5"],
#             "--optimizer": ["RMSprop"],
#             # "--padding": ["valid", "same"],
#             # "--phase_augment": [("None", "cutmix", "cutmix")],
#             "--save_model": ["True"],
#             "--scope": ["sub"],
#             "--seed": [6, 7, 8],
#             # "--spatial_dropout": [0, 0.1],
#         },
#         n_threads = 3    
#     )
# )

# runs.append(
#     RUN(
#         name="mixup",
#         args_combinations={
#             "--activation": ["relu"],
#             "--augment": [
#                 # None,
#                 # "cutmix",
#                 "mixup",
#                 # ("tailored", "cutmix", None, "cutmix", None),
#                 ],   
#             # "--batch_norm": ["False"],         
#             # "--conv_type": ["standard", "ds"],
#             "--decay": ["plateau"],
#             "--depth": [4],
#             # "--dropout": [0.1],
#             "--early_stopping": ["True"],
#             "--epochs": [400],
#             # "--fag": ["Flatten", "GAP"],
#             "--filters": [64],
#             # "--head": ["sigmoid"],
#             "--logdir_suffix": ["mixup"],
#             "--learning_rate": [0.1],
#             "--loss": ["KLD"],
#             "--model": ["model5"],
#             "--optimizer": ["RMSprop"],
#             # "--padding": ["valid", "same"],
#             # "--phase_augment": [("None", "cutmix", "cutmix")],
#             "--save_model": ["True"],
#             "--scope": ["sub"],
#             "--seed": [6, 7, 8],
#             # "--spatial_dropout": [0, 0.1],
#         },
#         n_threads = 3    
#     )
# )

if __name__ == "__main__":
    for run in runs:
        run.run()