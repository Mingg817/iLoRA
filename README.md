# iLoRA

#### Preparation

1. Prepare the environment:

```python
git clone
cd iLoRA
pip install -r requirements.txt
```

2. Prepare the pre-trained huggingface model of LLaMA2-7B (https://huggingface.co/meta-llama/Llama-2-7b-hf).
3. Download the data and checkpoints.
4. Prepare the data and checkpoints:

Put the data to the dir path data/ref/ and the checkpoints to the dir path checkpoints/.

#### Train iLoRA

Train iLoRA with a single A100 GPU on MovieLens dataset:

```python
sh train_movielens.sh
```

Train iLoRA with a single A100 GPU on Steam dataset:

```
sh train_steam.sh
```

Train iLoRA with a single A100 GPU on LastFM dataset:

```
sh train_lastfm.sh
```

Note that: set the `llm_path` argument with your own directory path of the Llama2 model.

##### Evaluate iLoRA

Test iLoRA with a single A100 GPU on MovieLens dataset:

```
sh test_movielens.sh
```

Test iLoRA with a single A100 GPU on Steam dataset:

```
sh test_steam.sh
```

Test iLoRA with a single A100 GPU on LastFM dataset:
