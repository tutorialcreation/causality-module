import pandas as pd

class EDA:
    def __init__(self) -> None:
        pass

    def data_checks(self,df: pd.DataFrame):
        """
        perform checks in the dataset

        args: 
            df (pd.DataFrame): the DataFrame which we are performing checks
        
        returns:
            dictionary of checks

        """
        checks = {
            "info":df.info(),
            "shape":df.shape,
            "uniqueness":df.apply(lambda x: len(x.unique())).sort_values(ascending=False).head(10),
            "missing_values":df.isnull().sum(),
            "duplicates":df.duplicated().sum(),

        }
        return checks
        