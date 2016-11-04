from landscape_simulator import *
import matplotlib.animation as animation
import types
import sys


def animate_landscape(landscape, steps, funding='best', individuals=False, dynamic=True, cutoff=0.7):
    """Runs the landscape simulator and creates an animation from the filled contour plot of each step.

    Requires ffmpeg."""
    fig = plt.figure()
    
    m = mgrid[:landscape.size, :landscape.size]
    
    ims = []
    for i in range(steps):
        im = plt.contourf(m[0], m[1], landscape.matrix, range(100), cmap='coolwarm')
        
        def setvisible(self, vis):
            for c in self.collections:
                c.set_visible(vis)
        im.set_visible = types.MethodType(setvisible, im, None)
        def setanimated(self, anim):
            for c in self.collections:
                c.set_animated(anim)
        im.set_animated = types.MethodType(setanimated, im, None)
        im.axes = plt.gca()
        if individuals:
            texts = []
            for ind in landscape.individuals:
                text = im.axes.text(ind[0]-0.5, ind[1]-0.5, str(ind[2]))
                texts.append(text)
        ims.append([im]+texts)
        landscape.step(funding=funding, dynamic=dynamic, cutoff=cutoff)
    ani = animation.ArtistAnimation(fig, ims, interval=5000, blit=False, repeat_delay=False)
    raw_input('press enter')
    ani.save('animate_sz%d_st%d_dynamic_%.1f_%s.mp4' % (landscape.size, steps, cutoff, funding),
             writer=animation.FFMpegFileWriter())
    
    plt.show()


if __name__ == '__main__':
    size = 50
    avg_countdown = 5
    steps = 100
    funding = 'lotto'
    dynamic = True
    cutoff = 1.1

    if len(sys.argv) > 1:
        funding = sys.argv[1]
    
    landscape = GaussianLandscape(size, size ** 2 / 100, 99)
    landscape.init_individuals(size ** 2 / 200, avg_countdown=avg_countdown)
    animate_landscape(landscape, steps, funding=funding, individuals=True, dynamic=dynamic, cutoff=cutoff)
