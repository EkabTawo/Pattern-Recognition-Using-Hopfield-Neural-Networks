
#####################################
#HOPFIELD NETWORK PREDICTIVE MODEL.
#####################################


##Importing the necessary libraries and packages in python.
from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


#Code to devlope each cell in  the grid for the GUI

class Cell():
    FILLED_COLOR_BG = "blue"
    EMPTY_COLOR_BG = "white"
    FILLED_COLOR_BORDER = "black"
    EMPTY_COLOR_BORDER = "black"

    def __init__(self, master, x, y, size):
        """ Constructor of the object called by Cell(...) """
        self.master = master
        self.abs = x
        self.ord = y
        self.size= size
        self.fill= False

    def _switch(self):
        """ Switch if the cell is filled or not. """
        self.fill = not self.fill

    def draw(self):
        """ order to the cell to draw its representation on the canvas """
        if self.master != None :
            fill = Cell.FILLED_COLOR_BG
            outline = Cell.FILLED_COLOR_BORDER

            if not self.fill:
                fill = Cell.EMPTY_COLOR_BG
                outline = Cell.EMPTY_COLOR_BORDER

            xmin = self.abs * self.size
            xmax = xmin + self.size
            ymin = self.ord * self.size
            ymax = ymin + self.size

            self.master.create_rectangle(xmin, ymin, xmax, ymax, fill = fill, outline = outline)


#Code to draw the grid which is used as the GUI.
class CellGrid(Canvas):
    def __init__(self,master, rowNumber, columnNumber, cellSize, *args, **kwargs):
        Canvas.__init__(self, master, width = cellSize * columnNumber , height = cellSize * rowNumber, *args, **kwargs)

        self.cellSize = cellSize

        self.grid = []
        for row in range(rowNumber):

            line = []
            for column in range(columnNumber):
                line.append(Cell(self, column, row, cellSize))

            self.grid.append(line)

        #memorize the cells that have been modified to avoid many switching of state during mouse motion.
        self.switched = []

        #bind click action
        self.bind("<Button-1>", self.handleMouseClick)  
        #bind moving while clicking
        self.bind("<B1-Motion>", self.handleMouseMotion)
        #bind release button action - clear the memory of midified cells.
        self.bind("<ButtonRelease-1>", lambda event: self.switched.clear())

        self.draw()


    def draw(self):
        for row in self.grid:
            for cell in row:
                cell.draw()

    def _eventCoords(self, event):
        row = int(event.y / self.cellSize)
        column = int(event.x / self.cellSize)
        return row, column

#Code used in computing the mouse click.
    def handleMouseClick(self, event):
        global net
        row, column = self._eventCoords(event)
        cell = self.grid[row][column]
        cell._switch()
        cell.draw()
        #add the cell to the list of cell switched during the click
        self.switched.append(cell)

        net[row,column] = (net[row,column] + 1)%2



    def handleMouseMotion(self, event):
        row, column = self._eventCoords(event)
        cell = self.grid[row][column]

        if cell not in self.switched:
            cell._switch()
            cell.draw()
            self.switched.append(cell)

            net[row,column] = (net[row,column] + 1)%2



if __name__ == "__main__" :

    def func(array):
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if array[i,j] == 0:
                    array[i,j] = -1
        return array

#Code for the output grid. 

    def output_grid(X,num='f'):
        rows = X.shape[0]
        cols = X.shape[1]

        fig, ax1 = plt.subplots()
        my_cmap = colors.ListedColormap(['white', 'blue'])
        ax1.matshow(np.reshape(X, (rows,cols)), cmap=my_cmap)

        x_ticks = np.arange(-0.5,cols)
        y_ticks = np.arange(-0.5,rows)

        ax1.set_xticks(y_ticks)
        ax1.set_yticks(y_ticks)
        ax1.grid(which='major', alpha=1, color='black', linewidth=1.5)

        for tick in ax1.xaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)

        for tick in ax1.yaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)

      
        fig.tight_layout()
        fig.savefig('grid_plot'+str(num)+'.png')



##########################
#Hopfield Network Model
##########################

##Assigning the number of  neurons as well as the rows and columns for the matrix.
    rows = 9
    cols = 9

    N = rows * cols
    stored_images = 50
    print('Give me ',stored_images, 'image(s)')

#Applying the Hebbian Learning based on the formula to devlope the matrix.  
#Updating the synaptic weights.
    W = np.zeros((N,N))
    for i in range(stored_images):
        net = np.zeros((rows,cols))
        app = Tk()
        grid = CellGrid(app, rows, cols, 50)
        grid.pack()
        app.mainloop()

        p = func(net)
        p = p.reshape(-1)
        p = p.reshape(N,1)
        W += p*p.T-np.identity(N)

    W = W/N
    
    print('')

    net = np.zeros((rows,cols))
    app = Tk()
    grid = CellGrid(app, rows, cols, 50)
    grid.pack()
    app.mainloop()

    print('')

##Calculation of the weighted sum to get the signs of the neurons.
    x = func(net)
    print(x)
    x = x.reshape(-1)
    x = x.reshape(N,1)

    z = np.zeros(N)
    for i in range(N):
        for j in range(N):
            if j != i:
                z[i] += W[i,j]*x[j]

    x = np.sign(z)

    print(z)
 
    print('')

    x = x.reshape(rows,cols)
    print(x)
    output_grid(x)
    
#####################
#####REFERENCES - https://github.com/andreasfelix/hopfieldnetwork/tree/main/examples/project4
#####################
