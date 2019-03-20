from code.manager import EvaluationManager

if __name__ == "__main__":
    evaluation_manager = EvaluationManager()
    evaluation_manager.evaluate('country_vectors.txt', tasks=['DocumentSimilarity'], debugging_mode = True)