import csv
import os
import pandas as pd

from evaluation_framework.SemanticAnalogies.semanticAnalogies_model import (
    SemanticAnalogiesModel as Model,
)
from evaluation_framework.abstract_taskManager import AbstractTaskManager
from numpy import mean
import numpy as np
from _collections import defaultdict
from pathlib2 import Path
from typing import List, Callable

task_name = "SemanticAnalogies"


class SemanticAnalogiesManager(AbstractTaskManager):
    """
    Manager of the semantic analogies task.
    """

    def __init__(
        self,
        data_manager,
        top_k: int,
        debugging_mode: bool,
        analogy_function: Callable[
            [np.ndarray, np.ndarray, np.ndarray], np.ndarray
        ] = None,
        datasets: List[str] = None,
    ):
        """Constructor. It initializes the manager of the semantic analogies task.

        Parameters
        ----------
        data_manager
            The data manager to read the dataset(s) and the input file with the vectors to evaluate.
        top_k : int
            the predicted vector is compared with all the vectors and the k nearest ones are depicted. If the actual
            vector is among the k nearest one, the task is considered correct.
        debugging_mode : bool
            TRUE to run the model by reporting all the errors and information; FALSE otherwise
        analogy_function : Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
            Is the function to compute the analogy. It takes 3 vectors and returns the predicted one.
        datasets : List[str]
            None if all datasets shall be evaluated. Specific datasets can also be named using this parameter.
        """
        super().__init__()
        self.debugging_mode = debugging_mode
        self.data_manager = data_manager
        self.analogy_function = analogy_function
        self.top_k = top_k
        self.datasets = datasets
        if debugging_mode:
            print("SemanticAnalogies task manager initialized")

    """
    It returns the task name.
    """

    @staticmethod
    def get_task_name():
        return task_name

    """
    It evaluates the Semantic analogies task.
    
    vectors: dataframe which contains the vectors data
    vector_file: path of the vector file
    vector_size: size of the vectors
    result_directory: directory where the results must be stored
    log_dictionary: dictionary to store all the information to store in the log file
    scores_dictionary: dictionary to store all the scores which will be used in the comparison phase
    """

    def evaluate(
        self,
        vectors,
        vector_file: str,
        vector_size,
        results_folder,
        log_dictionary,
        scores_dictionary,
    ):
        log_errors = ""

        vocab = self.data_manager.create_vocab(vectors, vector_file, vector_size)
        W_norm = self.data_manager.normalize_vectors(
            vectors, vector_file, vector_size, vocab
        )

        # check whether gold standard datasets have been passed through the constructor
        if self.datasets is not None:
            gold_standard_filenames = self.datasets
        else:
            gold_standard_filenames = self.get_gold_standard_file()

        scores = list()
        totalscores = defaultdict(list)

        for gold_standard_filename in gold_standard_filenames:
            gold_standard_file = SemanticAnalogiesManager.get_file_for_dataset(
                dataset=gold_standard_filename
            )

            data, ignored = self.data_manager.intersect_vectors_goldStandard(
                vectors, vector_file, vector_size, gold_standard_file
            )
            data_coverage = len(data) / (len(data) + len(ignored))

            self.storeIgnored(results_folder, gold_standard_filename, ignored)

            if len(data) == 0:
                log_errors += (
                    "SemanticAnalogies : Problems in merging vector with gold standard "
                    + gold_standard_filename
                    + "\n"
                )
                if self.debugging_mode:
                    print(
                        "SemanticAnalogies : Problems in merging vector with gold standard "
                        + gold_standard_filename
                    )
            else:
                model = Model(
                    task_name, self.top_k, self.debugging_mode, self.analogy_function
                )

                result = model.train(vocab, data, W_norm)
                result["gold_standard_file"] = gold_standard_filename
                result["coverage"] = data_coverage
                scores.append(result)

                totalscores[gold_standard_filename].append(result)

        self.storeResults(results_folder, scores)

        results_df = self.resultsAsDataFrame(totalscores)
        scores_dictionary[task_name] = results_df

        log_dictionary[task_name] = log_errors

    """
    It stores the entities which are in the dataset used as gold standard, but not in the input file.
    
    results_folder: directory where the results must be stored
    gold_standard_filename: the current dataset used as gold standard
    ignored: dataframe containing the ignored entities in the column NAME
    """

    def storeIgnored(self, results_folder, gold_standard_filename, ignored):
        if self.debugging_mode:
            print("Semantic analogies:" + str(len(ignored)) + " ignored quadruples")

        with open(
            results_folder
            + "/semanticAnalogies_"
            + gold_standard_filename
            + "_ignoredData.csv",
            "w",
        ) as file_result:
            fieldnames = ["entity_1", "entity_2", "entity_3", "entity_4"]
            writer = csv.DictWriter(file_result, fieldnames=fieldnames)
            writer.writeheader()

            for ignored_quadruplet in ignored:
                if self.debugging_mode:
                    print(
                        "Semantic analogies: Ignored quadruplet "
                        + str(ignored_quadruplet)
                    )

                writer.writerow(
                    {
                        "entity_1": ignored_quadruplet[0],
                        "entity_2": ignored_quadruplet[1],
                        "entity_3": ignored_quadruplet[2],
                        "entity_4": ignored_quadruplet[3],
                    }
                )

    """
    It stores the results of the Semantic Analogies task.
    
    results_folder: directory where the results must be stored
    gold_standard_filename: the current dataset used as gold standard
    scores: list of all the results returned by the model
    """

    def storeResults(self, results_folder, scores):
        with open(
            results_folder + "/semanticAnalogies_results.csv", "w"
        ) as file_result:
            fieldnames = [
                "task_name",
                "gold_standard_file",
                "coverage",
                "top_k_value",
                "right_answers",
                "tot_answers",
                "accuracy",
            ]
            writer = csv.DictWriter(file_result, fieldnames=fieldnames)
            writer.writeheader()
            for score in scores:
                writer.writerow(score)
                if self.debugging_mode:
                    print("SemanticAnalogies", score)

    """
    It converts the scores dictionary into a dataframe
    
    scores: list of all the results returned by the model
    """

    def resultsAsDataFrame(self, scores):
        data_dict = dict()
        data_dict["task_name"] = list()
        data_dict["gold_standard_file"] = list()
        data_dict["coverage"] = list()
        data_dict["model"] = list()
        data_dict["model_configuration"] = list()
        data_dict["metric"] = list()
        data_dict["score_value"] = list()

        metrics = self.get_metric_list()

        for (gold_standard_filename, gold_standard_scores) in scores.items():
            for metric in metrics:
                metric_scores = list()
                for score in gold_standard_scores:
                    metric_scores.append(score[metric])
                metric_score = mean(metric_scores)

                score = gold_standard_scores[0]

                data_dict["task_name"].append(score["task_name"])
                data_dict["gold_standard_file"].append(score["gold_standard_file"])
                data_dict["coverage"].append(score["coverage"])
                data_dict["model"].append("-")
                data_dict["model_configuration"].append("-")
                data_dict["metric"].append(metric)
                data_dict["score_value"].append(metric_score)

        results_df = pd.DataFrame(
            data_dict,
            columns=[
                "task_name",
                "gold_standard_file",
                "coverage",
                "model",
                "model_configuration",
                "metric",
                "score_value",
            ],
        )
        return results_df

    @staticmethod
    def get_file_for_dataset(dataset: str) -> str:
        """This method returns the absolute file path of a dataset.

        Parameters
        ----------
        dataset : str
            The dataset name for which the underlying file path shall be obtained.

        Returns
        -------
            The full path to the dataset file.
        """
        script_dir = os.path.dirname(__file__)
        rel_path = "data/" + dataset + ".txt"
        gold_standard_file = str(Path(os.path.join(script_dir, rel_path)))
        return gold_standard_file

    @staticmethod
    def get_gold_standard_file() -> List[str]:
        """It returns the dataset used as gold standard.

        Returns
        -------
            It returns the dataset used as gold standard.
        """
        return [
            "capital_country_entities",
            "all_capital_country_entities",
            "currency_entities",
            "city_state_entities",
        ]

    @staticmethod
    def get_metric_list() -> List[str]:
        """It returns the metrics used in the evaluation of the semantic analogies task.

        Returns
        -------
            It returns the metrics used in the evaluation of the semantic analogies task.
        """
        return ["accuracy"]
