from typing import Literal, Tuple, Optional, List
import numpy as np
from src.base import Element, Individual
import time
import warnings
from deap import tools
from copy import deepcopy
from src.mutations import *
from src.crossings import *
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

warnings.filterwarnings("ignore")

class MGGP:

    def __init__(self,
                 inputs: np.ndarray,
                 outputs: np.ndarray,
                 generations: int,
                 validation: Optional[Tuple[np.ndarray, np.ndarray]] = (None, None),
                 evaluationMode: Literal['RMSE', 'MSE', 'MAPE'] = 'RMSE',
                 evaluationType: Literal['OSA', 'MShooting', 'FreeRun'] = 'MShooting',
                 k: int = 5,
                 nTerms: int = 15,
                 maxHeight: int = 15,
                 weights: tuple = (-1,),
                 nDelays: int | List[int] = 15,
                 crossoverRate: float = 0.8,
                 mutationRate: float = 0.1,
                 populationSize: int = 100,
                 elitePercentage: int = 10
                 ):
        """
        Args:
            inputs (ndarray): The inputs in the system. Each column represent an input.
            outputs (ndarray): The outputs in the system. Each column represent an output.
            generations (int): Number of generations to train the model.
            validation (Optional[Tuple[np.ndarray, np.ndarray]]): Inputs and outputs to validate the model. Must be a tuple (inputs, outputs).
            evaluationMode (Literal['RMSE', 'MSE', 'MAPE']):  Mode to evaluate the models and ranking the better.
            evaluationType (Literal['OSA', 'MShooting', 'FreeRun']): One-Step-Ahead, Multiple-Shooting and Free-Run predictors.
            k (int): Used with Multiple-Shooting predictor. Define the number of shooting.
            nTerms (int): Number of terms each output model will possess.
            maxHeight (int): Maximum height of Genetic Program individual.
            weights (tuple): Defines the type of optimization (-1 for minimization, 1 for maximization). It must be a tuple.
            nDelays (float | Literal['fixed']): The number that will define the backshift operators q^{-n}, for n in delays.
            crossoverRate (float): Crossover probability.
            mutationRate (float): Mutation probability.
            populationSize (int): Population size.
            elitePercentage (int): Percentile of population to be kept in the next generation.
        """

        self.inputs = inputs
        self.outputs = outputs
        self.validation = validation
        self.nInputs = self.inputs.shape[1]
        self.nOutputs = self.outputs.shape[1]
        self.generations = generations
        self.evaluationMode = evaluationMode
        self.evaluationType = evaluationType

        self.crossoverRate = crossoverRate
        self.mutationRate = mutationRate
        self.populationSize = populationSize
        self.elitePercentage = elitePercentage
        self.k = k
        self.nTerms = nTerms
        self.weights = weights
        self.nDelays = nDelays
        self.maxHeight = maxHeight

        if self.evaluationMode not in ["MSE", "MAPE", "RMSE"]:
            raise Exception("Choose a measure between:\n" +
                            "MSE, MAPE, or RMSE")

        if self.nInputs > 1 and self.nOutputs == 1:
            self.mode = "MISO"
        elif self.nInputs > 1 and self.nOutputs > 1:
            self.mode = "MIMO"
        else:
            raise Exception("MGGP doesn't work with SISO systems")

        self.element = Element(weights=self.weights,
                               nDelays=self.nDelays,
                               nInputs=self.nInputs,
                               nOutputs=self.nOutputs,
                               nTerms=self.nTerms,
                               maxHeight=self.maxHeight,
                               mode=self.mode)

        self.element.renameArguments(self.buildArgumentsDict())

        self._toolbox = base.Toolbox()
        self._toolbox.register("evaluate", self.evaluation)

        self._mutList = []
        self._crossList = []
        self._stats = self._createStatistics()
        self._logbook = tools.Logbook()
        self._logbook.header = 'gen', 'evals', 'fitness'
        self._logbook.chapters['fitness'].header = 'min', 'avg', 'max'

        self._hofSize = int(round(self.populationSize * (self.elitePercentage / 100)))
        self._hof = tools.HallOfFame(self._hofSize)

        self._toolbox.register("select", tools.selTournament, tournsize=2)

        self.addMutation(MutGPOneTree)
        self.addMutation(MutGPUniform)
        self.addMutation(MutGPReplace)

        self.addCrossOver(CrossHighUniform)
        self.addCrossOver(CrossLowUniform)

    def addMutation(self, mutation):
        self._mutList.append(mutation(self.element))

    def addCrossOver(self, crossover):
        self._crossList.append(crossover(self.element))

    def _delAttr(self, ind):
        try:
            del ind.fitness.values
            del ind.funcs
            del ind.kfuncs
            del ind.lagMax
        except AttributeError:
            pass


    def stream(self):
        print(self._logbook.stream)

    def initPop(self, seed=[]):
        if len(seed) > self.populationSize: raise Exception('Seed exceeds population size!')
        if seed == []:
            self._pop = self.element._toolbox.population(self.populationSize)
        else:
            self._pop = self.element._toolbox.population(self.populationSize - len(seed))
            self._pop += seed
        invalid_ind = [ind for ind in self._pop if not ind.fitness.valid]

        if self.evaluationType == 'OSA':
            fitnesses = list(tqdm(self._toolbox.map(self._toolbox.evaluate, invalid_ind), total=len(invalid_ind), desc="Evaluating Initial Population"))

        else:
            with ProcessPoolExecutor(max_workers=14) as executor:
                fitnesses = list(tqdm(executor.map(self.evaluation, invalid_ind), total=len(invalid_ind), desc="Evaluating Initial Population"))

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        record = {'fitness': self._stats.compile(self._pop)}
        self._logbook.record(gen=0, evals=len(invalid_ind), **record)
        # print(self._logbook)
        self._hof.update(self._pop)

    def get_fitness_value(self, individual):
        return individual.fitness.values[0]

    def _createStatistics(self):
        stats = tools.Statistics(self.get_fitness_value)
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        stats.register("min", np.min)
        return stats

    def step(self, gen_number):
        if not self._pop:
            raise Exception('Population must be initialized!')

        offspring = [deepcopy(ind) for ind in self._toolbox.select(self._pop, self.populationSize - self._hofSize)]

        for i in range(0, len(offspring) - 1, 2):
            if np.random.random() < self.crossoverRate:
                cross = random.choice(self._crossList)
                offspring[i], offspring[i + 1] = cross.cross(offspring[i], offspring[i + 1])
                self._delAttr(offspring[i])
                self._delAttr(offspring[i + 1])

        for i in range(len(offspring)):
            if np.random.random() < self.mutationRate:
                mut = random.choice(self._mutList)
                offspring[i], = mut.mutate(offspring[i])
                self._delAttr(offspring[i])

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        if self.evaluationType == 'OSA':
            fitnesses = list(tqdm(self._toolbox.map(self._toolbox.evaluate, invalid_ind), total=len(invalid_ind),
                                  desc="Evaluating Population"))

        else:
            with ProcessPoolExecutor(max_workers=14) as executor:
                fitnesses = list(tqdm(executor.map(self.evaluation, invalid_ind), total=len(invalid_ind),
                                      desc="Evaluating Population"))

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit


        self._pop = self._hof.items + offspring
        self._hof.update(self._pop)

        #---Record--Statistics-----------------------------------------------------
        record = {'fitness': self._stats.compile(self._pop)}

        self._logbook.record(gen=gen_number+1, evals=len(invalid_ind), **record)


    def buildArgumentsDict(self) -> dict:
        arguments = dict()
        arguments.update({f'ARG{i}': f'y{i + 1}' for i in range(self.nOutputs)})
        arguments.update({f'ARG{self.nOutputs + i}': f'u{i + 1}' for i in range(self.nInputs)})

        return arguments

    # @staticmethod
    def evaluation(self, ind: Individual) -> tuple[float]:
        try:

            # if len(ind._theta) == 0:
            self.element.compileModel(ind)
            theta_value = ind.leastSquares(self.outputs, self.inputs)
            ind._theta = theta_value

            if self.evaluationType == "MShooting":
                yp, yd = ind.predict("MShooting", self.k, self.outputs, self.inputs)
            else:
                yp, yd = ind.predict(self.evaluationType, self.outputs, self.inputs)
            error = ind.score(yd, yp, self.evaluationMode)
            return error,

        except np.linalg.LinAlgError:
            return (np.inf,)

    def run(self) -> None:
        # pool = multiprocessing.Pool(6)  # using 4 processor cores
        # self.evolver._toolbox.register("map", pool.map)
        self._toolbox.register("map", map)

        init = time.time()
        self.initPop()
        self.stream()

        for g in range(self.generations):
            self.step(g)
            self.stream()

        model = self._hof[0]
        self.element.compileModel(model)
        theta_value = model.leastSquares(self.outputs, self.inputs)
        model._theta = list(theta_value)
        # print(model)
        print(model.to_equation())
        # print(model._theta)

        if all([value is not None for value in self.validation]):
            u_val, y_val = self.validation

            if u_val.shape[1] != self.nInputs:
                raise Exception("the number os inputs to validate and to train, must have the same length")

            if y_val.shape[1] != self.nOutputs:
                raise Exception("the number os outputs to validate and to train, must have the same length")

            if self.evaluationType == "MShooting":
                yp, yd = model.predict("MShooting", self.k, y_val, u_val)
            else:
                yp, yd = model.predict(self.evaluationType, self.outputs, self.inputs)
            error = round(model.score(yd, yp, self.evaluationMode), 6)
            print(f"{self.evaluationMode} in validation dataset: {error}")

        end = time.time()
        print(f"Executed in: {round(end - init, 3)} seg")