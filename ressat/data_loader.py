import os
import pickle
import numpy as np
import pandas as pd


# This section split is provided as an example workflow and can be modified by users according to their own experimental design.
def load_sections(data_dir, section_num):
    """
    Return (train_sections, val_sections, test_sections).

    If section_num >= 3:
        Section_1 is used as the test section.
        Section_2 is used as the validation section.
        Section_3 and later sections are used as training sections.

    If section_num == 2:
        Section_1 is used as the test section.
        Section_2 is used as the training section.
        No validation section is used.
    """
    sections = []
    for i in range(section_num):
        data_path = os.path.join(data_dir, f"Section_{i+1}")
        with open(os.path.join(data_path, "dataset.pkl"), "rb") as f:
            d = pickle.load(f)
        with open(os.path.join(data_path, "locations.pkl"), "rb") as f:
            locs = pickle.load(f)
        sections.append({
            "data": d,
            "locs": locs,
            "section_id": i + 1,
        })

    if len(sections) >= 3:
        test_sections = [sections[0]]
        val_sections = [sections[1]]
        train_sections = sections[2:]
    elif len(sections) == 2:
        test_sections = [sections[0]]
        val_sections = None
        train_sections = [sections[1]]
    else:
        raise ValueError("Need at least 2 sections.")

    print(f"Train sections: {', '.join(str(s['section_id']) for s in train_sections)}")

    if val_sections is not None:
        print(f"Val sections  : {', '.join(str(s['section_id']) for s in val_sections)}")

    print(f"Test sections : {', '.join(str(s['section_id']) for s in test_sections)}")

    return train_sections, val_sections, test_sections

def load_gene_names(data_dir, gene_file="gene_list.csv", gene_col="gene"):
    gene_path = os.path.join(data_dir, gene_file)
    gene_names = pd.read_csv(gene_path)[gene_col].tolist()
    print(f"Predicting {len(gene_names)} Genes ...")
    return gene_names
