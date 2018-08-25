import pandas as pd
import os


fnames_example = [
    "data/perturbations/dog/2018-07-22 15:39:08.328737/scores.csv",
    "data/perturbations/cat/2018-07-22 15:23:36.936765/scores.csv",
    "data/perturbations/cat2/2018-07-22 15:31:22.377249/scores.csv",
    "data/perturbations/car/2018-07-22 15:15:42.316242/scores.csv",
    "data/perturbations/excavator/2018-07-22 15:46:53.904665/scores.csv",
    #     "data/perturbations/image_admixture/2018-07-22 15:54:54.637076/",
    "data/perturbations/palace/2018-07-22 16:02:56.023002/scores.csv",
]


def main(fnames=[],
         save_to=".test_table.tex"):
    # if just folders are given, add "scores.csv" to the path
    fnames = [x if x.endswith("scores.csv") else os.path.join(x, "scores.csv") for x in fnames]
    full_df = build_full_df(fnames)

    with open(save_to, "w") as fp:
        fp.write(full_df.to_latex())

    return full_df


def process_class_name(x):
    names = x.split(",")
    name = names[0]
    pref_names = ["cat", "dog", "car", "vehicle"]
    for other_name in names:
        for pref_name in pref_names:
            if pref_name in other_name:
                name = other_name
                break
    words = name.split(" ")
    #     if len(words) > 2: words.insert(2, "\n")
    return " ".join(words)


def process_class_score_df(df):
    df.columns = ["class", "score"]
    df = df.sort_values("score", ascending=False)
    df = df.head()
    df["class"] = df["class"].map(process_class_name)
    df["score"] = df["score"].round(decimals=4)
    return df


def get_sample_subdf(fname="data/perturbations/dog/2018-07-22 15:39:08.328737/scores.csv"):
    sample_df = pd.read_csv(fname)
    sample_df.columns = ["index", "class", "original", "perturbated"]

    df_orig = sample_df[["class", "original"]]
    df_orig = process_class_score_df(df_orig)
    df_pert = sample_df[["class", "perturbated"]]
    df_pert = process_class_score_df(df_pert)
    df_orig.columns = pd.MultiIndex.from_product([["original"], ["class", "score"]])
    df_orig.index = range(5)
    df_pert.columns = pd.MultiIndex.from_product([["perturbated"], ["class", "score"]])
    df_pert.index = range(5)
    row_df = pd.concat([df_orig, df_pert], axis=1)

    # add empty row at the end:
    empty = pd.Series(['', '', '', ''], index=row_df.columns)
    row_df = row_df.append(empty, ignore_index=True)

    return row_df


def build_full_df(fnames):
    sub_dfs = []
    for sample_index, fname in enumerate(fnames):
        sample_df = get_sample_subdf(fname)
        #         sample_df.index = pd.MultiIndex.from_product([[sample_index+1], range(1, 6)],
        #                                                     names=["sample", "class rank"])
        #         sample_df.index = 5 * [sample_index + 1]
        sample_df.index = pd.MultiIndex.from_product([[sample_index + 1], 6 * [""]],  # 5 +  one empty row
                                                     names=["sample", ""])
        sub_dfs.append(sample_df)
    df = pd.concat(sub_dfs, axis=0)
    df.index.name = "sample"
    return df


if __name__ == "__main__":
    main()

