from typing import Literal
import numpy as np
from base import Element
from evolvers import EvolDefault
import multiprocessing
import time


class MGGP:

    def __init__(self,
                 inputs: np.ndarray,
                 outputs: np.ndarray,
                 generations: int,
                 evaluationMode: Literal['RMSE', 'MSE', 'MAPE'] = 'RMSE',
                 evaluationType: Literal['OSA', 'MShooting', 'FreeRun'] = 'MShooting',
                 k: int = 5,
                 nTerms: int = 15,
                 maxHeight: int = 15,
                 weights: tuple = (-1,),
                 nDelays: float | Literal['fixed'] = 15,
                 crossoverRate: float = 0.8,
                 mutationRate: float = 0.1,
                 populationSize: int = 100,
                 elitePercentage: int = 10
                 ):
        self.inputs = inputs
        self.outputs = outputs
        self.nInputs = self.inputs.shape[1]
        self.nOutputs = self.outputs.shape[1]
        self.generations = generations
        self.evaluationMode = evaluationMode
        self.evaluationType = evaluationType
        if self.evaluationMode not in ["MSE", "MAPE", "RMSE"]:
            raise Exception("Choose a measure between:\n" +
                            "MSE, MAPE, or RMSE")
        self.crossoverRate = crossoverRate
        self.mutationRate = mutationRate
        self.populationSize = populationSize
        self.elitePercentage = elitePercentage
        self.k = k
        self.nTerms = nTerms
        self.weights = weights
        self.nDelays = nDelays
        self.maxHeight = maxHeight

        if self.nInputs > 1 and self.nOutputs == 1:
            self.mode = "MISO"
        elif self.nInputs > 1 and self.nOutputs > 1:
            self.mode = "MIMO"

        self.element = Element(weights=self.weights,
                               nDelays=self.nDelays,
                               nInputs=self.nInputs,
                               nOutputs=self.nOutputs,
                               nTerms=self.nTerms,
                               maxHeight=self.maxHeight,
                               mode=self.mode)

        self.element.renameArguments(self.buildArgumentsDict())

        self.evolver = EvolDefault(element=self.element,
                                   evaluate=self.evaluation,
                                   popSize=self.populationSize,
                                   elitePerc=self.elitePercentage,
                                   CXPB=self.crossoverRate,
                                   MTPB=self.mutationRate)

    def buildArgumentsDict(self) -> dict:
        arguments = dict()
        arguments.update({f'ARG{i}': f'y{i + 1}' for i in range(self.nOutputs)})
        arguments.update({f'ARG{self.nOutputs + i}': f'u{i + 1}' for i in range(self.nInputs)})

        return arguments

    # @staticmethod
    def evaluation(self, ind):
        try:
            self.element.compileModel(ind)
            theta_value = ind.leastSquares(self.outputs, self.inputs)
            ind._theta = list(theta_value)
            # yp, yd = ind.predict("OSA", y, u)
            # yp, yd = ind.predict("FreeRun", y, u)

            if self.evaluationType == "MShooting":
                yp, yd = ind.predict("MShooting", self.k, self.outputs, self.inputs)
            else:
                yp, yd = ind.predict(self.evaluationType, self.outputs, self.inputs)
            error = ind.score(yd, yp, self.evaluationMode)
            return error,

        except np.linalg.LinAlgError:
            return (np.inf,)

    def run(self):
        #pool = multiprocessing.Pool(6)  # using 4 processor cores
        #self.evolver._toolbox.register("map", pool.map)
        self.evolver._toolbox.register("map", map)

        init = time.time()
        self.evolver.initPop()
        self.evolver.stream()

        for g in range(self.generations):
            self.evolver.step()
            self.evolver.stream()

        end = time.time()
        hof = self.evolver.getHof()
        model = hof[0]

        self.element.compileModel(model)
        theta_value = model.leastSquares(self.outputs, self.inputs)
        model._theta = list(theta_value)
        print(model)
        # print(model.to_equation())
        print(model._theta)
        print(f"time: {round(end - init, 3)} seg")
