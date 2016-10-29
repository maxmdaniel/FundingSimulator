from landscape_simulator import *
import matplotlib.animation as animation
import types
import sys

def animate_landscape(landscape, steps, funding='best', individuals=False, dynamic=True):
    """Runs the landscape simulator and creates an animation from the filled contour plot of each step.

    Requires ffmpeg."""
    fig = plt.figure()
    
    m = mgrid[:landscape.size, :landscape.size]
    
    ims = []
    for i in range(steps):
        im = plt.contourf(m[0], m[1], landscape.matrix, range(landscape.size*2), cmap='gist_earth')
        
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
                text = im.axes.text(ind[0], ind[1], str(ind[2]))
                texts.append(text)
        ims.append([im]+texts)
        landscape.step(funding=funding, dynamic=dynamic)
    ani = animation.ArtistAnimation(fig, ims, interval=2000, blit=False, repeat_delay=False)
    raw_input('press enter')
    ani.save('animate_sz%d_st%d_dynamic_%s.mp4' % (landscape.size, steps, funding), writer=animation.FFMpegFileWriter())
    
    plt.show()


if __name__ == '__main__':
    size = 50
    avg_countdown = 5
    steps = 50
    funding = 'lotto'
    dynamic = True

    if len(sys.argv) > 1:
        funding = sys.argv[1]
    
    landscape = GaussianLandscape(size, size / 2, (size - 1) * 2)
    landscape.init_individuals(int(size ** 0.75), avg_countdown=avg_countdown)
    animate_landscape(landscape, steps, funding=funding, individuals=True, dynamic=dynamic)
