# Script to visualize the results of a multi-objective optimization (Pareto front)

import datetime
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import sys

from mpl_toolkits.mplot3d import Axes3D # this is for 3D figures

sns.set_theme() # set the SeaBorn graphical theme, it looks nice 

def plot_2d(df, output_folder, fitness_a, fitness_b) :

    # set the data
    fitness_a_values = df[fitness_a].values
    fitness_b_values = df[fitness_b].values

    # plot the figure
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1)

    ax.scatter(fitness_a_values, fitness_b_values, color='blue', alpha=0.7) # alpha sets transparency

    ax.set_title("Pareto front, %s vs %s" % (fitness_a, fitness_b))
    ax.set_xlabel(fitness_a)
    ax.set_ylabel(fitness_b)

    plt.savefig(os.path.join(output_folder, "pareto-front-" + fitness_a + "-" + fitness_b + ".png"), dpi=300)
    plt.close(fig)

    return

def main() :

    # a few hard-coded values
    #input_file = "pareto-front.csv"
    input_file = "pareto-front_2.csv"
    fitness_names = ["Economic_Impact", "Environmental_Impact", "Social_Impact"] # these should be changed with the names of the corresponding columns in the CSV file
    # output folder with a unique name, using the current date and time
    output_folder = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+ "-visualize-results" 

    # read the CSV file
    print("Reading file \"%s\"..." % input_file)
    df = pd.read_csv(input_file)
    print(df)

    # create an output folder, if it does not already exist
    if not os.path.exists(output_folder) : os.makedirs(output_folder)

    # let's create some visualizations!
    plot_2d(df, output_folder, fitness_names[0], fitness_names[1])
    plot_2d(df, output_folder, fitness_names[1], fitness_names[2])
    plot_2d(df, output_folder, fitness_names[0], fitness_names[2])

    # also, a 3d plot with everything
    fig = plt.figure(figsize=(8,8))
    fig = plt.figure()
    ax = Axes3D(fig)

    x = df[fitness_names[0]].values
    y = df[fitness_names[1]].values
    z = df[fitness_names[2]].values

    ax.scatter(x, y, z, c=x, marker='o', alpha=0.7)
    ax.set_xlabel(fitness_names[0])
    ax.set_ylabel(fitness_names[0])
    ax.set_zlabel(fitness_names[0])

    plt.savefig(os.path.join(output_folder, "3dplot.png"), dpi=300)
    plt.show()

    return

if __name__ == "__main__" :
    sys.exit( main() )
