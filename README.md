## This repo is the implementation of Multi-task Bert using self-supervised technique 

This model will uitilize the dataset that have multiple labels. It will have n+1-heads according to n-tasks and a **MaskedLM** head.

Our method achieves amazing result with our **NEU**, **VSFC** and **ViHSD** datasets:

| Task | Accuracy | F1 macro | F1 weighted |
|----------|----------|----------|----------|
|  NEU sentiment   | 84.42	|85.15|	84.43
|  NEU classification   | 81.33 |	73.98 |	81.57
|  VSFC sentiment        |93.94	|83.77|	94.19
|  VSFC topic  | 89.45 |	80.82	|90.15
|  ViHSD  |88.31 |	68.49 |	88.77


To train the model, modify the model config in `train.py` and run 

`python3 train.py`

We made a website for the implementation of the model, you can checkout [here](https://frontend-nlp.vercel.app/)

If you are seeing this, it means that we havent finished documenting our code. Please be patient
