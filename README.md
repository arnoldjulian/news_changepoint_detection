# Changepoint detection

This repository contains the implementation of the Confusion method
from the publication "Machine learning change points in real-world news data".

To install all dependencies, run

```bash
pip install .[dev]
```

You will need to download the spacy language models as well:

```bash
python -m spacy download en
```

## Running trainings

To run one type of training run

```bash
make train L=... dataset=<dataset_type>  model=<model_type>
```

In the Makefile you can find exactly what configuration files are used for each training.

## Running evaluations and generating summaries

Running an evaluation on a single training generates all the metrics required to evaluate a model. The summarization
script takes individual results and generates comparative metrics (the top-n delta from the paper).

Example evaluation:

```bash
make eval L=8 dataset=generated model=confusion
```

It is also possible to run evaluation for the baseline Total Variation Distance model (no previous training required):

```bash
make eval_tvd L=8 dataset=generated
```

Example summarization:

```bash
make summarize dataset=generated
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For any further questions, email me at:  
**csabazsolnai (at) pm (dot) me**
