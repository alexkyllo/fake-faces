"""labeling.py
utility for labeling test images manually."""
import os
import random
import csv
import click
import questionary
import logging
from PIL import Image

LABEL_FILE = "labels.csv"

AGES = [
    "0-2",
    "3-9",
    "10-19",
    "20-29",
    "30-39",
    "40-49",
    "50-59",
    "60-69",
    "more than 70",
    "unsure",
]

GENDERS = [
    "Female",
    "Male",
    "unsure",
]

RACES = [
    "East Asian",
    "White",
    "Latino_Hispanic",
    "Southeast Asian",
    "Black",
    "Indian",
    "Middle Eastern",
    "unsure",
]


def get_random_file(path):
    """return a random file from a directory."""
    return os.path.join(path, random.choice(os.listdir(path)))


def save_labels(img_path, age, gender, race, csv_path=LABEL_FILE):
    """Save the provided labels by appending to a CSV file."""
    fieldnames = ["file", "age", "gender", "race"]
    dict_data = [{"file": img_path, "age": age, "gender": gender, "race": race}]
    if os.path.isfile(csv_path):
        with open(csv_path, "a") as label_file:
            writer = csv.DictWriter(label_file, fieldnames=fieldnames)
            for row in dict_data:
                writer.writerow(row)
    else:
        with open(csv_path, "w") as label_file:
            writer = csv.DictWriter(label_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in dict_data:
                writer.writerow(row)


@click.command()
@click.argument("path", type=click.Path(exists=True))
def label(path):
    """Show a random face image file from PATH and prompt for labels."""
    logger = logging.getLogger(__name__)
    while True:
        print("Displaying a random image for labeling...")
        img_path = get_random_file(path)
        img = Image.open(img_path)
        img.show()
        age = questionary.select(
            "What age range is this person?",
            AGES,
        ).unsafe_ask()
        gender = questionary.select(
            "What gender is this person?",
            GENDERS,
        ).unsafe_ask()
        race = questionary.select(
            "What race is this person?",
            RACES,
        ).unsafe_ask()
        save_labels(img_path, age, gender, race, LABEL_FILE)
        print(f"Labels appended to {LABEL_FILE}")
