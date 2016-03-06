from landscape_simulator import GaussianLandscape
from numpy import arange
from matplotlib import pyplot as plt


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

if __name__ == '__main__':
    plot_landscape()
