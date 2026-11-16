# InternVL training

The [training guide](../docs/internvl-reproduction.md) gives the recorded environment, answer-SFT and refusal-SFT recipes, and runnable commands. Run commands from the repository root.

`ivl_recipe.py` constructs the training rows; `ivl_common.py` implements model operations and LoRA; `ivl_train_sft.py` trains each stage; `ivl_merge.py` merges and checks the checkpoint. `recorded_run.json` contains the data, training and merge records.
