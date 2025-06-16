"""Tools for generating artificial articles with LLMs."""

import argparse
import os.path

import pandas as pd
import yaml
from openai import OpenAI
from tqdm import tqdm

from topic_transition.utils import get_dates_for_interval

PROMPT_TEMPLATE = """
I would like you to write an imaginary news article about the following topic: {topic}
The imaginary article was published in the '{section}' section of the Guardian newspaper.

I want the output to be in the following format:
Title: <Title>
Main text: <Main text>

Please don't output anything else than this.

I will also provide a real-world Guardian article with its title and main text as a reference. Please try to use
as many topics, people, and locations from it in your article as possible.
Here is the real article:
{article}
"""


def main(configuration: str) -> None:
    """Generate artificial articles based on existing Guardian articles and specified topics."""

    def extract_text(chat_completion):
        content = chat_completion.choices[0].message.content
        if "Main text:" not in content:
            raise ValueError("Incorrect formatting")
        if "Title:" not in content:
            raise ValueError("Incorrect formatting")
        title, text = content.split("Main text:")
        title = title[len("Title:") :].strip()  # noqa: E203
        text = text.strip()
        return title, text

    with open(configuration, "r") as file:
        config = yaml.safe_load(file)
    articles_per_day = int(config["articles_per_day"])
    start_date = config["start_date"]
    end_date = config["end_date"]
    split_date = config["split_date"]
    assert start_date <= split_date <= end_date
    reference_df = pd.read_pickle(config["reference_dataset"])
    reference_df["full_text"] = reference_df["webTitle"] + "\n" + reference_df["text"]
    year, filename = config["reference_dataset"].split(os.path.sep)[-2:]
    section_id = filename[:-4]
    dates = get_dates_for_interval(start_date, end_date)
    data: dict[str, list] = {"sectionId": [], "webPublicationDate": [], "webTitle": [], "text": []}
    client = OpenAI(
        api_key=config["openai_api_key"],
    )
    main_topic = config["main_topic_1"]
    for dt in tqdm(dates):
        if dt >= split_date:
            main_topic = config["main_topic_2"]
        for _ in range(articles_per_day):
            while True:
                reference_article = reference_df.sample(n=1).iloc[0]
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": PROMPT_TEMPLATE.format(
                                topic=main_topic,
                                section=reference_article["sectionId"],
                                article=reference_article["full_text"],
                            ),
                        }
                    ],
                    model="gpt-3.5-turbo",
                )
                try:
                    title, text = extract_text(chat_completion)
                    break
                except ValueError:
                    continue
            data["sectionId"].append(section_id)
            data["webPublicationDate"].append(dt)
            data["webTitle"].append(title)
            data["text"].append(text)

    df = pd.DataFrame.from_dict(data)
    out_path = os.path.join(config["output_base"], year)
    os.makedirs(out_path, exist_ok=True)
    out_path = os.path.join(out_path, filename)
    df.to_pickle(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("configuration")
    args = parser.parse_args()

    main(args.configuration)
