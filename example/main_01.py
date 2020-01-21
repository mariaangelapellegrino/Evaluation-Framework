from evaluation_framework.manager import FrameworkManager

if __name__ == "__main__":
    evaluation_manager = FrameworkManager()
    evaluation_manager.evaluate('objectFrequencyS.h5', vector_file_format='hdf5',  debugging_mode = False)