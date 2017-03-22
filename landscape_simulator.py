from numpy import *
from numpy.random import *
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import random


class BasicLandscape(object):
    """Base class for landscape simulation.

    The basic landscape is initialised with random values in the range [0, 1].
    Individuals are seeded in random locations and perform countdown-delayed local hill climbing.
    Individuals contribute to accumulated significance and collective vision.
    The landscape itself is static, and there is no selection of individuals.
    """

    def __init__(self, size):
        self.size = size
        self.matrix = rand((size, size))
        self.vision = zeros((size, size))

    def generate_countdowns(self, num=1):
        """Generate num integer countdowns, evenly distributed between [avg_countdown / 2, avg_countdown * 1.5]. """
        return randint(max(1, self.avg_countdown/2), int(self.avg_countdown*1.5)+1, size=(num, 1))

    def generate_individuals(self, num):
        """Generate num individuals at random positions with the avg_countdown set by init_individuals."""
        return hstack((
            randint(self.size-1, size=(num, 2)),  # starting positions.
            self.generate_countdowns(num)  # countdowns.
        ))
    
    def set_individual_vision(self, ind):
        """Update the vision matrix to include the 3*3 area around the individual."""
        x = ind[0]
        y = ind[1]
        for xi in range(-self.agent_vision, self.agent_vision+1):
            for yi in range(-self.agent_vision, self.agent_vision+1):
                if (x+xi < self.size) and (y+yi < self.size) and (x+xi >= 0) and (y+yi >= 0):
                    self.vision[[x+xi], [y+yi]] = 1

    def init_individuals(self, num, avg_countdown=3, agent_vision=1):
        """Initialise num individuals on the landscape and set their vision."""
        self.avg_countdown = avg_countdown
        self.agent_vision = agent_vision
        self.individuals = self.generate_individuals(num)
        self.accumulated_significance = 0
        self.step_significance_contributions = []
        for ind in self.individuals:
            self.set_individual_vision(ind)
    
    def print_matrix(self):
        print self.matrix
    
    def plot(self, individuals=False):
        """Plot a gray-scale filled contour map of the landscape, optionally with individual counters at locations."""
        m = mgrid[:self.size, :self.size]
        plt.contourf(m[0], m[1], self.matrix, range(self.size*2), cmap=plt.cm.gray)
        if individuals:
            for ind in self.individuals:
                plt.text(ind[0], ind[1], str(ind[2]))
            
    def show(self, individuals=False):
        """Show the result of plot."""
        self.plot(individuals=individuals)
        plt.colorbar()
        plt.show()

    def save(self, filename, individuals=False):
        """Save the result of plot to a file."""
        self.plot(individuals=individuals)
        plt.colorbar()
        plt.savefig(filename, bbox_inches=0)
        plt.show()

    def plot_vision(self):
        """Plot the current vision matrix as black/white tiles."""
        m = mgrid[:self.size,:self.size]
        plt.pcolormesh(m[0],m[1],self.vision,cmap=plt.cm.hot)
    
    def show_vision(self):
        """Show the result of plot_vision."""
        self.plot_vision()
        plt.show()
    
    def countdown_individual(self,ind):
        """Reduce all countdowns by 1, generating new countdowns for those that reached zero."""
        ind[2] -= 1  # reduce countdown
        if ind[2]:  # countdown not zero
            return ind, -1  # new_z as -1 indicates the individual hasn't moved
        
        # set new countdown
        ind[2] = self.generate_countdowns(1)

        # contribute to accumulated significance based on current location
        ind_z = self.matrix[ind[0], ind[1]]
        self.accumulated_significance += ind_z
        self.step_significance_contributions.append(ind_z)
        
        return ind, ind_z
        
    def move_individual(self,ind):
        """Move an individual to the highest point in the local 3*3 area."""
        end_x = start_x = ind[0]
        end_y = start_y = ind[1]
        
        max_z = self.matrix[start_x, start_y]
        
        # move to highest neighbour (including current position)
        for x_offset in range(-1, 2):
            cur_x = min(self.size-1, max(0, start_x+x_offset))
            for y_offset in range(-1, 2):
                cur_y = min(self.size-1, max(0, start_y+y_offset))
                if self.matrix[cur_x, cur_y] > max_z:
                    max_z = self.matrix[cur_x, cur_y]
                    end_x = cur_x
                    end_y = cur_y
        ind[0] = end_x
        ind[1] = end_y
        
        self.set_individual_vision(ind)
        
        return ind

    def step(self):
        """Countdown all individuals and move the ones whose countdown reached zero."""
        for i in range(len(self.individuals)):
            self.individuals[i], ind_z = self.countdown_individual(self.individuals[i])
            if ind_z != -1:
                self.individuals[i] = self.move_individual(self.individuals[i])


def get_peak(size, loc, sig, max_height):
    """Returns a 2d matrix of size*size with a peak centred at loc, with width sig, of max_height."""
    m = mgrid[:size, :size]
    biv = mlab.bivariate_normal(m[0], m[1], sig[0], sig[1], loc[0], loc[1])
    return biv*float(max_height)/biv.max()


class GaussianLandscape(BasicLandscape):
    """A feature-rich landscape simulator.

    The Gaussian Landscape is initialised by adding bivariate Gaussian hills to a flat landscape (hence the name).
    Funding:
        The Gaussian Landscape adds a simulation of funding by selecting from a pool of candidate individuals a subset
        that will explore the landscape. Various selection mechanisms are modelled (see step).
    Dynamic significance:
        The Gaussian Landscape can be deformed as a result of exploration. Various dynamic processes are modelled.
    """

    def __init__(self, size, num_peaks, max_height):
        self.size = size
        self.vision = zeros((size, size))
        self.matrix = zeros((size, size))
        self.max_height = max_height
        peaks = randint(1, self.size-1, size=(num_peaks, 3))  # randomly generate loc and sig for each peak.
        for peak in peaks:
            self.add_gaussian(
                (peak[0], peak[1]),  # location of the peak on the landscape
                peak[2],  # width of the peak along x and y
                random.random()  # peak height
            )
        self.matrix *= max_height / self.matrix.max()  # scale landscape to match max_height.

    def add_gaussian(self, loc, sig, height):  # height can be negative
        self.matrix += get_peak(self.size, loc, (sig, sig), height)
        # do not allow negative values
        self.matrix = self.matrix.clip(min=0, max=self.max_height)

    def winner_takes_all(self, ind):
        """A dynamic effect, sets significance to zero at individual location (once explored)."""
        loc = [ind[0], ind[1]]
        height = self.matrix[loc[0], loc[1]] * -1  # lower current position by its height
        sig = self.size * 0.01
        self.add_gaussian(loc, sig, height)

    def reduced_novelty(self, ind):
        """A dynamic effect, lower nearby significance (once a significant location is explored)."""
        loc = [ind[0], ind[1]]
        height = self.matrix[loc[0], loc[1]] * -0.5  # lower local neighbourhood by majority of height
        sig = self.size * 0.05
        self.add_gaussian(loc, sig, height)

    def new_avenue(self):
        """A dynamic effect, add a peak at a random location (once a significant location is explored)."""
        loc = randint(self.size-1, size=(2, 1))
        height = randint(self.matrix.max())
        sig = self.size * 0.06
        self.add_gaussian(loc, sig, height)
        
    def step(self, cutoff=0.7, funding='best', dynamic=True):
        """Run through a single simulation step.

        Each simulation step:
        1. Countdown all individuals.
        2. Accumulate significance from individuals who reached zero countdown.
        3. If a simulation with dynamic significance, alter the landscape with dymanic processes.
        4. Generate candidate pool from zero-countdown individuals and new individuals.
        5. Select individuals from the candidate pool based on the funding method.
        6. Add vision for selected individuals.
        """

        self.step_significance_contributions = []

        individual_indexes_to_move = []
        for i in range(len(self.individuals)):
            # update individual's countdown, check if research finished
            self.individuals[i], ind_z = self.countdown_individual(self.individuals[i])

            # Simulate dynamic processes
            if ind_z != -1:
                if dynamic:
                    # Effects triggered above cutoff
                    if (float(ind_z)/self.matrix.max()) > cutoff:
                        self.reduced_novelty(self.individuals[i])
                        self.new_avenue()

                    # Always triggered effects
                    self.winner_takes_all(self.individuals[i])
                
                individual_indexes_to_move.append(i)
        
        # move individual on new landscape
        for i in individual_indexes_to_move:
            self.individuals[i] = self.move_individual(self.individuals[i])
        
        # --- Funding stage ---
        if not individual_indexes_to_move:  # if there are no candidates there is no free cash
            return
        
        num_total_individuals = len(self.individuals)
        
        # remove moved individuals from current individuals, add them to candidate list
        old_candidates = []
        for i in individual_indexes_to_move:
            old_candidates.append(self.individuals[i])
        self.individuals = delete(self.individuals, individual_indexes_to_move, 0)
        
        # add new candidates until number of applicants == total num of individuals
        new_individuals = self.generate_individuals(num_total_individuals - len(individual_indexes_to_move))
        candidates = vstack((array(old_candidates), new_individuals))
        
        if funding == 'best':  # Select candidates at highest positions, regardless of vision.
            zs = [self.matrix[ind[0], ind[1]] for ind in candidates]
            zs.sort()
            zs.reverse()
            zs = zs[:len(individual_indexes_to_move)]
            
            count = 0
            for ind in candidates:
                if self.matrix[ind[0], ind[1]] in zs:
                    self.individuals = vstack((self.individuals, ind))
                    self.set_individual_vision(ind)
                    count += 1
                    if count >= len(individual_indexes_to_move):
                        break

        elif funding == 'best_visible':  # Select highest, only from visible positions (ignore invisible).
            non_visible_candidate_indexes = []
            for i in range(len(candidates)):
                ind = candidates[i]
                if not self.vision[ind[0], ind[1]]:
                    non_visible_candidate_indexes.append(i)
            candidates = delete(candidates, non_visible_candidate_indexes, 0)
            zs = [self.matrix[ind[0], ind[1]] for ind in candidates]
            zs.sort()
            zs.reverse()
            zs = zs[:len(individual_indexes_to_move)]
            
            count = 0
            for ind in candidates:
                if self.matrix[ind[0], ind[1]] in zs:
                    self.individuals = vstack((self.individuals, ind))
                    self.set_individual_vision(ind)
                    count += 1
                    if count >= len(individual_indexes_to_move):
                        break

        elif funding == 'lotto':  # Select candidates at random, regardless of height or vision.
            shuffle(candidates)
            candidates = candidates[:len(individual_indexes_to_move)]
            for ind in candidates:
                self.set_individual_vision(ind)
            self.individuals = vstack((self.individuals, candidates))

        elif funding == 'triage':  # Select half by best_visible and half by lotto.
            non_visible_candidate_indexes = []
            for i in range(len(candidates)):
                ind = candidates[i]
                if not self.vision[ind[0], ind[1]]:
                    non_visible_candidate_indexes.append(i)
            
            # Use lotto on half the candidates
            non_visible_candidates = candidates[non_visible_candidate_indexes]
            if non_visible_candidates.size:  # For long-running simulations there may be none.
                shuffle(non_visible_candidates)
            non_visible_candidates = non_visible_candidates[:len(individual_indexes_to_move)/2]
            for ind in non_visible_candidates:
                self.set_individual_vision(ind)
            self.individuals = vstack((self.individuals, non_visible_candidates))
            
            num_remaining = len(individual_indexes_to_move) - len(non_visible_candidates)
            
            # Use best_visible on the remainder
            visible_candidates = delete(candidates, non_visible_candidate_indexes, 0)
            zs = [self.matrix[ind[0], ind[1]] for ind in visible_candidates]
            zs.sort()
            zs.reverse()
            zs = zs[:num_remaining]
            count = 0
            for ind in candidates:
                if self.matrix[ind[0], ind[1]] in zs:
                    self.individuals = vstack((self.individuals, ind))
                    self.set_individual_vision(ind)
                    count += 1
                    if count >= num_remaining:
                        break

        elif funding == 'oldboys':  # No new candidates, zero-countdown individuals continue (no selection).
            self.individuals = vstack((self.individuals, array(old_candidates)))
        else:
            raise KeyError('Unknown funding option %s' % str(funding))

        # All selection methods keep the active number of individuals fixed.
        assert(len(self.individuals) == num_total_individuals)

if __name__ == '__main__':
    size = 100
    num_peaks = 50
    max_height = 200
    num_individuals = 30
    avg_countdown = 5

    landscape = GaussianLandscape(size, num_peaks, max_height)
    landscape.init_individuals(num_individuals, avg_countdown)

    landscape.show(individuals=True)
