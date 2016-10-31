from landscape_simulator import GaussianLandscape
from numpy import arange, mean, std
from matplotlib import pyplot as plt
import itertools


def show_landscapes(landscapes, individuals=False):
    """Shows a plot of multiple landscapes as subplots."""
    plt.figure()
    for i in range(len(landscapes)):
        plt.subplot(len(landscapes)*100+10+i)
        landscapes[i].plot(individuals=individuals)
    plt.show()


def plot_landscape(funding_options=('best', 'best_visible', 'lotto', 'triage', 'oldboys'),
                   size=50, num_steps=50, avg_countdown=5, num_runs=5, dynamic=True):
    """Plot statistics about accumulated significance for a simulated Gaussian Landscape.

    If plotting a single funding method, will show the accumulation of significance over time.
    If plotting a comparison of several methods, will show both a comparison of accumulation over time and a bar chart
    (with error bars) comparing accumulated significance at the end of the simulation.
    """
    if len(funding_options) > 1:
        all_lines = []
        for run in range(num_runs):  # runs the simulation several times for averaging.
            fitness_lines = []
            for funding in funding_options:
                landscape = GaussianLandscape(size, size/2, (size-1)*2)
                landscape.init_individuals(int(size**0.75), avg_countdown=avg_countdown)
                fitness_steps = []
                for i in range(num_steps):
                    if i % 10 == 0:
                        print funding, i
                    landscape.step(funding=funding, dynamic=dynamic)
                    fitness_steps.append(landscape.accumulated_significance)
                fitness_lines.append([funding, fitness_steps])
            all_lines.append(fitness_lines)
        if num_runs != 1:
            # Calculates average values and standard error.
            fitness_lines = []
            fitness_errors = []
            for f_i in range(len(funding_options)):
                fitness_line = []
                fitness_error = []
                for step in range(num_steps):
                    fitness_line.append(sum([l[f_i][1][step] for l in all_lines]) / num_runs)
                fitness_lines.append([funding_options[f_i], fitness_line])
                for step in range(num_steps):
                    err_sqr = [(l[f_i][1][step] - fitness_line[step]) ** 2 for l in all_lines]
                    fitness_error.append((sum(err_sqr) / (num_runs - 1)) ** 0.5)
                fitness_errors.append([funding_options[f_i], fitness_error])

        # Plot accumulated significance over time.
        r = range(num_steps)
        for fl in fitness_lines:
            plt.plot(r, fl[1], label=fl[0])
        plt.legend(loc='upper left')
        plt.xlabel('Simulation steps')
        plt.ylabel('Accumulated significance')
        plt.title('Size: %d, Steps: %d, Avg. Countdown: %d' % (size, num_steps, avg_countdown))
        plt.show()

        # Plot comparison bar chart with error bars
        plt.bar(arange(len(funding_options)),
                [l[1][-1] for l in fitness_lines],
                1,
                color='gray',
                ecolor='black',
                yerr=[e[1][-1] for e in fitness_errors])
        plt.ylabel('Accumulated significance')
        plt.xticks(arange(len(funding_options)) + 0.5, funding_options)
        plt.title('Size: %d, Steps: %d, Avg. Countdown: %d' % (size, num_steps, avg_countdown))

        plt.show()
        return

    if len(funding_options) == 1:
        funding = funding_options[0]

        landscape = GaussianLandscape(size, size/2, (size-1)*2)
        landscape.init_individuals(int(size**0.75), avg_countdown=avg_countdown)

        fitness_steps = []
        for i in range(num_steps):
            landscape.step(funding=funding, dynamic=dynamic)
            fitness_steps.append(landscape.accumulated_significance)

        plt.plot(fitness_steps)
        plt.xlabel('Simulation steps')
        plt.ylabel('Accumulated significance')
        plt.title('Size: %d, Steps: %d, Avg. Countdown: %d'%(size,num_steps,avg_countdown))
        plt.show()
        return

    else:
        raise ValueError('Must have at least one funding option')


def plot_variation(funding_options=('best', 'best_visible', 'lotto', 'oldboys'),
                   size=50,
                   num_peaks = None,
                   max_height = 99,
                   num_agents=None,
                   num_steps=50,
                   avg_countdown=5,
                   num_runs=5,
                   dynamic=True):
    """Plot accumulated significance at simulation end as a dependant variable of some model parameter.

    Shows performance at accumulating significance at the end of the simulation for each funding option,
    as a function of variation in some model parameter (given as a list of values to explore).
    """
    model_params = {
        'size': size,
        'num_peaks': num_peaks,
        'max_height': max_height,
        'num_agents': num_agents,
        'avg_countdown': avg_countdown
    }

    independent_var_name = None
    for k in model_params:
        if isinstance(model_params[k], list):
            independent_var_name = k
    if independent_var_name is None:
        raise ValueError('At least one model parameter must be a list')

    results = {f: ([], [], []) for f in funding_options}
    for v in model_params[independent_var_name]:
        model_param = model_params.copy()
        model_param[independent_var_name] = v
        # Defaults that are relative to the landscape size
        if model_param['num_peaks'] is None:
            model_param['num_peaks'] = model_param['size']**2 / 100
        if model_param['num_agents'] is None:
            model_param['num_agents'] = model_param['size']**2 / 200

        all_methods = {f: [] for f in funding_options}
        for run in range(num_runs):  # runs the simulation several times for averaging.
            for funding in funding_options:
                landscape = GaussianLandscape(
                    model_param['size'],
                    model_param['num_peaks'],
                    model_param['max_height'])
                landscape.init_individuals(
                    model_param['num_agents'],
                    avg_countdown=model_param['avg_countdown'])
                for i in range(num_steps):
                    if i % 10 == 0:
                        print funding, i
                    landscape.step(funding=funding, dynamic=dynamic)
                all_methods[funding].append(landscape.accumulated_significance)
        # Calculates average values and standard deviation.
        for f in funding_options:
            results[f][0].append(v)
            results[f][1].append(mean(all_methods[f]))
            results[f][2].append(std(all_methods[f]))

    # Plot accumulated significance per variable value.
    formats = itertools.cycle(['-o', '--s', '-.+', ':*'])
    for f, res in results.iteritems():
        plt.errorbar(res[0], res[1], label=f, yerr=res[2], fmt=formats.next())
    plt.xlim([0, int(max(model_params[independent_var_name])*1.1)])
    plt.legend(loc='upper left')
    plt.xlabel(independent_var_name.replace('_', ' ').title())
    plt.ylabel('Accumulated Significance')
    plt.show()

    return

if __name__ == '__main__':
    plot_landscape()
