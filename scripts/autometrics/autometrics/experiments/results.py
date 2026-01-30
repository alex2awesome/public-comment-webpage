import pandas as pd
import json
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import plotly.graph_objects as go

class Result(ABC):
    @abstractmethod
    def save(self, output_dir: str, name: str):
        print(f"NOT saving results to {output_dir}/{name}.  Please implement this method.")
        pass

class TabularResult(Result):
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def save(self, output_dir: str, name: str):
        self.dataframe.to_csv(f"{output_dir}/{name}.csv")   

class JSONResult(Result):
    def __init__(self, json_dict: dict):
        self.json_dict = json_dict

    def save(self, output_dir: str, name: str):
        with open(f"{output_dir}/{name}.json", "w") as f:
            json.dump(self.json_dict, f)

class TextResult(Result):
    def __init__(self, text: str):
        self.text = text

    def save(self, output_dir: str, name: str):
        with open(f"{output_dir}/{name}.txt", "w") as f:
            f.write(self.text)

class FigureResult(Result):
    def __init__(self, figure: plt.Figure):
        self.figure = figure

    def save(self, output_dir: str, name: str):
        self.figure.savefig(f"{output_dir}/{name}.pdf")

class PlotlyResult(Result):
    def __init__(self, figure: go.Figure):
        self.figure = figure

    def save(self, output_dir: str, name: str):
        self.figure.write_html(f"{output_dir}/{name}.html")