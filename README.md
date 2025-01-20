# iLoRA

#### Preparation

1. Prepare the environment:

```python
git clone
cd iLoRA
pip install -r requirements.txt
```

2. Prepare the pre-trained huggingface model of Llama2-7B (https://huggingface.co/meta-llama/Llama-2-7b-hf).
3. Modify the paths inside the .sh file.


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

##### For the environmental issues mentioned by everyone during the reproduction process, we have attempted to help resolve them and have listed some solutions:

If you encounter an error with your transformers/generation/utils.py file, please replace it with the debug/utils.py file we have provided in your environment.

If you encounter an error with your transformers/models/llama/modeling_llama.py file, please replace it with the debug/modeling_llama.py file.

Thank you all for your attention to our work! Wishing you success in your research.

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

```
sh test_lastfm.sh
```
