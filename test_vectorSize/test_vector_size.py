from evaluation_framework.manager import FrameworkManager
import pandas as pd

vectors_file = {}


def generate_cropped_files():
    vector_filename = "uniform_classification_regression.txt"

    vec_size = 200

    headers = ["name"]
    for i in range(0, vec_size + 1):
        headers.append(i)

    vectors_df = pd.read_csv(
        vector_filename, "\s+", names=headers, encoding="utf-8", index_col=False
    )

    store_path = "test_vectors/"

    crop = [10, 20, 50, 100, 150, 180, 200]

    for break_index in crop:
        headers = ["name"]
        for i in range(0, break_index + 1):
            headers.append(i)

        filename = store_path + "vectors_" + str(break_index) + ".csv"
        vectors_df.loc[:, headers].to_csv(
            filename,
            sep=" ",
            encoding="utf-8",
            index=False,
            header=False,
            float_format="%.15f",
        )

        vectors_file[filename] = break_index


def evaluate_cropped_files():
    evaluation_manager = FrameworkManager()

    for filename in vectors_file:
        vector_size = vectors_file[filename]

        evaluation_manager.evaluate(
            filename,
            vector_size=vector_size,
            tasks=["Regression"],
            parallel=True,
            debugging_mode=False,
        )


generate_cropped_files()
evaluate_cropped_files()
