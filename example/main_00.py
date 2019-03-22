from code.manager import FrameworkManager

if __name__ == "__main__":
    evaluation_manager = FrameworkManager()
    evaluation_manager.evaluate('country_vectors.txt', parallel=True,  debugging_mode = False)