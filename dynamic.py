import numpy as np
import lmfit
import re
import scipy


def get_nb_levels(model):
    """ Take a model object and extract the number of level used

    @param (lmfit.Parameters) model: The input model

    @return (int): The number of levels
    """
    regexp = re.compile('^k_(\d+)_(\d+)')
    nb = -1
    for key in model:
        match = regexp.search(key)
        if match:
            i, j = int(match.group(1)), int(match.group(2))
            maxi = max(i, j)
            nb = max(nb, maxi)
    return nb+1


def compute_rates(model):
    """ Take a model object and compute the rate matrix Kij

    @param (lmfit.Parameters) model: The input model

    @return (np.array): The n*n rate matrix Kij of the rates from level i to level j
    """
    size = get_nb_levels(model)
    rates = np.zeros((size,size))
    regexp = re.compile('^k_(\d+)_(\d+)')
    for key in model:
        match = regexp.search(key)
        if match:
            i, j = int(match.group(1)), int(match.group(2))
            rates[j, i] = model[key].value
            rates[i, i] -= rates[j, i]
    return rates


def simulate(initial_state, times, model):
    """ Take a model, an initial state, and a list of times at which to compute the state

    @param (np.array) initial_state: The vector of size (n) representing the populations of each levels at time 0
    @param (np.array) times: The vector of times (of length L) at which the state must be computed
    @param (lmfit.Parameters) model: The input model

    @return (np.array): The states at each time as a (L, n) array

    This method compute the evolution of the state under a constant regime (defined by the model object)
    """
    rates = compute_rates(model)
    lamdas, P = np.linalg.eig(rates)
    P_inverse = np.linalg.inv(P)
    result = []
    for t in times:
        r = P.dot(np.diag(np.exp(t*lamdas)).dot(P_inverse))
        population = r.dot(initial_state)
        result.append(population)
    result = np.array(result)
    return result


def simulate_sequence(model, dt, sequence, initial_state=None, get_from_recursive=True, **arguments):
    """ Simulate a laser sequence based on a model and a sequence object that describe the variation of the parameters

    @param (lmfit.Parameters) model: The input model
    @param (float) dt: The resolution in time to simulate. This parameters do not impact computation fidelity.
    @param (list) sequence: A list of object representing the sequence applied to the system
    @param (np.array) initial_state: The vector of size (n) representing the populations of each levels at time 0
    @param (bool) get_from_recursive: If true, the initial state is computed as the last state of a first iteration
    @params (dict) **arguments: Parameters of the sequence as key value.

    @return tuple(np.array, np.array): The populations at every dt interval (M, n),
                                       The values of the varying parameters (ex: 'power') over time

    Example sequence :
        [(550e-9, {'power':0}),
         (6e-6, {'power': 'laser'}),
         (1e-6, {'power':0}),
         'pi',
         (1e-6, {'power':0}),
         (6e-6, {'power': 'laser'}),
         (1.5e-6, {'power':0})]

    Example arguments
     arguments : {'laser' : 1e-3 }


    Here model['power'].value will be set from the value corresponding to 'laser' in the params, here 1e-3

    ---

    If no initial_state is provided, the default value is all the population in level 0.

    ---

    The string 'pi' in the sequence can be used to flip the population of level 0 and 1 with an efficiency given by the
    parameter model['pi_efficiency'].

    """
    if initial_state is None and get_from_recursive is False:
        initial_state = np.zeros(get_nb_levels(model))
        initial_state[0] = 1
    if initial_state is None and get_from_recursive is True:
        first_call = simulate_sequence(model, dt, sequence, initial_state=None, get_from_recursive=False, **arguments)
        populations, _ = first_call
        initial_state = populations[-1, :]
    populations = []
    keys_vs_time_arrays = {}
    for regime in sequence:
        if isinstance(regime, tuple):
            length, parameters = regime
            times = np.arange(length / dt) * dt
            for key in parameters:
                value = parameters[key]
                if isinstance(value, str):
                    value = arguments[value]
                model[key].value = value
                if key not in keys_vs_time_arrays.keys():
                    keys_vs_time_arrays[key] = []
                keys_vs_time_arrays[key].append(times*0+value)
            res = simulate(initial_state, times, model)
            populations.append(res)
            initial_state = res[-1, :]
        if regime == 'pi':
            i0, i1, eff = initial_state[0], initial_state[1], model['pi_efficiency'].value
            initial_state[0] = eff * i1 + (1 - eff) * i0
            initial_state[1] = eff * i0 + (1 - eff) * i1
    keys_vs_time_arrays = {key: np.concatenate(keys_vs_time_arrays[key]) for key in keys_vs_time_arrays.keys()}
    return np.concatenate(populations), keys_vs_time_arrays


# Example model :

# model = lmfit.Parameters()
# model.add('excitation_rate_over_power', value=9e+10)
# model.add('pi_efficiency', value=0.9)
# model.add('factor', value=1075858.95)
# model.add('radiative_rate', value=65.9e6)
# model.add('bg_over_power', value=5e+04/3e-03)  # from saturation curve
#
# model.add('power', value=1)
#
# model.add('k_0_2', expr='excitation_rate_over_power*power')
# model.add('k_1_3', expr='excitation_rate_over_power*power')
# model.add('k_2_0', expr='radiative_rate')
# model.add('k_3_1', expr='radiative_rate')
# model.add('k_2_4', value=11.4e6)
# model.add('k_3_4', value=92.1e6)
# model.add('k_4_0', value=4.87e6)
# model.add('k_4_1', value=2.35e6)


