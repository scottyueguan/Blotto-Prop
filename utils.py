from operator import itemgetter


class Vertices:
    def __init__(self, vertices, connections):
        self.vertices = vertices
        self.connections = connections

    def plot(self, ax, color, legend):
        vertices = self.vertices
        xdata = [vertices[i][0] for i in range(len(vertices))]
        ydata = [vertices[i][1] for i in range(len(vertices))]
        zdata = [vertices[i][2] for i in range(len(vertices))]

        ax.scatter3D(xdata, ydata, zdata, color=color)

        if self.connections is not None:
            for i, connection in enumerate(self.connections):
                start = connection[0]
                end = connection[1]
                xline = [vertices[start][0], vertices[end][0]]
                yline = [vertices[start][1], vertices[end][1]]
                zline = [vertices[start][2], vertices[end][2]]

                if i == 0:
                    ax.plot3D(xline, yline, zline, color + "-", label=legend)
                else:
                    ax.plot3D(xline, yline, zline, color + "-")

        ax.legend(loc='lower center')

        return ax