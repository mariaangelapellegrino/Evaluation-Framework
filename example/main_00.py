from evaluation_framework.manager import FrameworkManager

if __name__ == "__main__":
    evaluation_manager = FrameworkManager()
    #evaluation_manager.evaluate('uniform_classification_regression.txt', parallel=True,  debugging_mode = False)
    evaluation_manager.evaluate('uniform_classification_regression.txt', tasks=['Classification', 'Regression'], parallel=True,  debugging_mode = False)