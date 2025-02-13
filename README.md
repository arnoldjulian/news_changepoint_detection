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

For example, to run trainings on all available generated datasets with the transformer model:

```bash
make train L=10 dataset=generated model=ff_embed
```

For the trainings with the real data with artificial split:

```bash
make train L=180 dataset=artificial_split model=ff_embed
```

For the real world data:

```bash
make train L=180 dataset=guardian_extended model=ff_embed
```

In the Makefile you can find exactly what configuration files are used for each training.
In order to run the trainings, you will need to put the training data in the appropriate directories like so:

Repository Root  
└── dvc  
    └── datasets  
        ├── artificial_split  
        ├── 2000  
        │   └── us-news_uk-news_2000-04-14.pkl  
        │   ...  
        ├── generated  
        └── guardian_extended  

## Running evaluations and generating summaries

Running an evaluation on a single training generates all the metrics required to evaluate a model. The summarization
script takes individual results and generates comparative metrics (the top-n delta from the paper).

Example evaluation:

```bash
make eval L=10 dataset=generated model=ff_embed
```

It is also possible to run evaluation for the baseline Total Variation Distance model (no previous training required):

```bash
make eval_tvd L=10 dataset=generated
```

Example summarization:

```bash
make summarize dataset=generated
```

For the evaluations, you will need the events .csv files,
TODO: add Zenodo link


## Reproducing metrics from the paper

In order to help reproduce the results, I included a file called
```
reproduce.txt
```
that contains all commands that you should need to run to reproduce our results.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For any further questions, email me at:  
**csabazsolnai (at) pm (dot) me**
