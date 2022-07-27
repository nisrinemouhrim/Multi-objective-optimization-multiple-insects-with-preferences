# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10
 20:13:54 2022

@author: nisri
"""

# Script to perform multi-objective optimization of an insect chain, using only one type of insect

import copy
import inspyred # library for evolutionary algorithms
import io
import numpy as np
import os
import pandas as pd # to manipulate CSV files
import random # to generate random numbers
import sys
import math

from random import randint
from json import load, dump
from random import randrange


def load_instance(json_file):
    """
    Inputs: path to json file
    Outputs: json file object if it exists, or else returns NoneType
    """
    if os.path.exists(path=json_file):
        with io.open(json_file, 'rt', newline='') as file_object:
            return load(file_object)
        
    print("Error: cannot read file %s" % json_file)
    return None

# this function calculate the distance between GPS points 
def haversine(coord1, coord2):
    R = 6372800  # Earth radius in meters
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    
    phi1, phi2 = math.radians(lat1), math.radians(lat2) 
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lon2 - lon1)
    
    a = math.sin(dphi/2)**2 + \
        math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1 - a))

# this function just converts an individual from the internal representation to the external representation
def convert_individual_to_values(candidate, boundaries) :

    SC = candidate["SC"]
    Nl = int(candidate["Nl"] * (boundaries["Nl"][SC][1] - boundaries["Nl"][SC][0]) + boundaries["Nl"][SC][0])
    AIF = candidate["AIF"] * (boundaries["AIF"][SC][1] - boundaries["AIF"][SC][0]) + boundaries["AIF"][SC][0]
    F = candidate["F"]
    EQ = candidate["EQ"]
    C = candidate["C"]
    r = candidate["r"] 
    theta = candidate["theta"]
    # P = candidate["P"]            
    RW = candidate["RW"] * (boundaries["RW"][C-1][1] - boundaries["RW"][C-1][0]) + boundaries["RW"][C-1][0]
    I = candidate["I"]
    
    

        
        

    return SC, Nl, AIF, F, EQ, RW, I, C, r, theta

def fitness_function(candidate, json_instance, boundaries) :  

    operating_profit = 0 # maximize
    insect_frass = 0 # minimize
    labor_safety = 0 # maximize    
    
    SC, Nl, AIF, F, EQ, RW, I, C, r, theta = convert_individual_to_values(candidate, boundaries)
    
    # operating profit and frass
    biomass = 0.0

    insect_frass = 0.0
    labor_safety = 0.0
    labor_safety_cost= 0.0
    labor_safety_max = 0.0
    FWP_max = 0.0
    social_rejection = 0.0
    profit = 0.0
    
    
    # equipment cost
    #weight = random.uniform(0, 1) # TODO why was this weight randomized?

    social_rejection_max = 0.0
    
    for index, equip_dict in enumerate(json_instance["equipments"]) : 
        labor_safety_cost += equip_dict["equipment_cost"] * EQ[index] * Nl
        labor_safety += equip_dict["equipment_cost"] * EQ[index] * json_instance["SFls"][SC-1] * Nl
        labor_safety_max += equip_dict["equipment_cost"] * json_instance["SFls"][SC-1] * Nl
    
    
    DFS, DC =  Distances(candidate, json_instance, boundaries)
    
    FWP = (RW / json_instance["countries"][C-1]["RWT"])* (json_instance["countries"][C-1]["CWT"]/json_instance["countries"][C-1]["MLW"])*(1 - np.square(json_instance["countries"][C-1]["IEF"]))         
    FWP_max = (boundaries["RW"][C-1][1] / json_instance["countries"][C-1]["RWT"])* (json_instance["countries"][C-1]["CWT"]/json_instance["countries"][C-1]["MLW"])*(1 - np.square(json_instance["countries"][C-1]["IEF"]))
    for index_2, insect_dict in enumerate(json_instance["insects"]) : 
        social_rejection += json_instance["countries"][C-1]["pop"] / DC[C-1] * insect_dict["risk"] * I[index_2] * SC  
        social_rejection_max += json_instance["countries"][C-1]["pop"] / DC[C-1] * 4 * 1 * SC
    
    social_aspect = labor_safety / labor_safety_max + FWP / FWP_max +  social_rejection/social_rejection_max
    
    for index, insect_dict in enumerate(json_instance["insects"]) :
        for index_2, feed_dict in enumerate(insect_dict["feed"]) :
            biomass = AIF * F[index_2] * feed_dict["FCE"]
            insect_frass += AIF / feed_dict["FCE"] * (1.0 - feed_dict["FCE"]) * json_instance["Frsf"][SC-1]
            profit += (insect_dict["sales_price"] - feed_dict["Costs"]) * biomass
    operating_profit =  profit - RW*Nl*12 -labor_safety_cost 
    

    environmental_aspect = max(json_instance["lambda"][0] * (max(DFS) - DFS[C-2]) / (max(DFS) - min(DFS)),  json_instance["lambda"][1] * insect_frass)
    
    return operating_profit, 1/environmental_aspect, social_aspect


    


def generator(random, args) :

    boundaries = args["boundaries"]
    json_instance = args["json_instance"]
    preferences = args["preferences"]
    
    
    # here we need to generate a random individual and check that the boundaries are respected
    # to (hopefully) make our lives easier, individuals are encoded as dictionaries
    print("Generating new individual...")
    individual = dict()

    # first step: randomize the scale of the company and the real wages
    
    individual["SC"] = random.choice(preferences["Sc"])

    # other values are in (0,1), and will then be scaled depending on the scale of the company (before evaluation)
    individual["AIF"] = random.uniform(0, 1)
    individual["Nl"] = random.uniform(0, 1)
    individual["RW"] = random.uniform(0, 1)

    # protective equipments can or cannot be acquired
    individual["EQ"] = list()
    for i in range(0, boundaries["EQ"]) :
        individual["EQ"].append(random.choice([0, 1]))

    # types of feed: they are encoded as a list of floats, that has to be normalized (they represent percentages)
    individual["F"] = list()
    for i in range(0, boundaries["F"]) :
        individual["F"].append(random.uniform(0, 1) * preferences["fd"][i])
    denominator = sum(individual["F"])
    

    for i in range(0, boundaries["F"]) :
        individual["F"][i] /= denominator
        
    # an insect can or cannot be chosen
    individual["I"] = list()
    start = True
    while any(a != 0 for a in individual["I"]) == False or len(individual["I"]) == 0 or start == True:
        individual["I"].clear()
        for i in range(0, boundaries["I"]) :
            individual["I"].append(random.choice([0, 1]) * preferences["I"][i]) 
        start = False

    # a country should be chosen
    individual["C"] = random.choice(preferences["C"]) 
    
    # the polar coordinate of the farm position should be generated as a couple of float
    individual["r"] =  random.uniform(0, 1)
    individual["theta"] =  random.uniform(0, 1)
        
    return individual



def evaluator(candidates, args) :

    json_instance = args["json_instance"]
    boundaries = args["boundaries"]

    list_of_fitness_values = []
    for candidate in candidates :
        f1, f2, f3 = fitness_function(candidate, json_instance, boundaries)
        list_of_fitness_values.append(inspyred.ec.emo.Pareto( [f1, f2, f3] )) # in this case, for multi-objective optimization we need to create a Pareto fitness object with a list of values
    
    return list_of_fitness_values

def Distances(candidate, json_instance, boundaries) :
    
    list_of_distance = []
    distances_to_center = []
     
    for index, countries_dict in enumerate(json_instance['countries']):
        distance_farm_feed_suppliers = 0.0
        for index_3, coordonate_dict in enumerate(countries_dict["city_center_location"]):
            x, y = convert_coordinate_gps_cart(coordonate_dict["lat"],coordonate_dict["lon"])
        r = candidate["r"] * countries_dict["radius"]
        theta = candidate["theta"] * 360      
        X, Y = convert_coordinate_polar_cart(x, y, r, theta)
        coord_1 = {X,Y}
        coord_2 = {x, y}
        distances_to_center.append(distance_between_two_cart_points(coord_1, coord_2))
        for index_2, feed_suppliers_dict in enumerate(json_instance['countries'][index]["feed_suppliers"]):
            coord_2 = convert_coordinate_gps_cart(feed_suppliers_dict["lat"],feed_suppliers_dict["lon"])
            distance_farm_feed_suppliers += distance_between_two_cart_points(coord_1, coord_2 )  
        list_of_distance.append(distance_farm_feed_suppliers)  

    return list_of_distance, distances_to_center
    

def convert_coordinate_polar_cart(x, y, r, theta):
    """
    this function converts polar coordinates to catesian coordinates
    """ 
    x_ = x + r * math.cos(theta)
    y_ = y + r * math.sin(theta)  
    
    return x_, y_

def convert_coordinate_gps_cart(lat, lon):
    """
    this function converts gps coordinates to cartisian coordinates
    """   
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    R = 6371 # radius of the earth
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    
    return x, y

def distance_between_two_cart_points(A, B):
    
    A = list(A)
    B = list(B)   
    distance = math.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)
    
    return distance



def random_lat_lon(lat, lon, min_radius, max_radius ):
    """
    this function produces an array with pairs lat, lon
    """   
    point =[lat, lon] 
    lat = (point[0] * math.pi / 180)
    lon = (point[1] * math.pi / 180)
    earth_radius = 6371000
    distance = math.sqrt(random.uniform(0, 1) * (pow(max_radius, 2) - pow(min_radius, 2)) + pow(min_radius ,2))
    delta_lat = math.cos(random.uniform(0, 1) * (math.pi)) * distance / earth_radius
    sign = random.randint(0,2) * 2 - 1
    delta_lon = sign * math.acos(
    ((math.cos(distance/earth_radius) - math.cos(delta_lat)) /
    (math.cos(lat) * math.cos(delta_lat + lat))) + 1)

    result = dict()
    result["lat"] = (lat + delta_lat) * 180 / math.pi
    result["lon"] = (lon + delta_lon) * 180 / math.pi
    
    return result



@inspyred.ec.variators.crossover
def variator(random, candidate1, candidate2, args) :
    
    children = []
    boundaries = args["boundaries"]
    preferences = args["preferences"]

    # decide whether we are going to perform a cross-over
    perform_crossover = random.uniform(0, 1) < 0.8 # this is True or False
    # cross-over
    if perform_crossover :
        print("I am going to perform a cross-over!")
        child1 = copy.deepcopy(candidate1)
        child2 = copy.deepcopy(candidate2)

        # for every key in the dictionary (every part of the individual)
        # randomly swap everything with a 50% probability
        # TODO maybe do something better with values associated to lists (EQ, F) ?
        for k in child1 :
            if random.uniform(0, 1) < 0.5 :
                temp = child1[k]
                child1[k] = child2[k]
                child2[k] = temp

        # append children
        children = [child1, child2]     

    # mutation(s)
    # if there are no children, create a new one
    if len(children) == 0 :
        children.append(copy.deepcopy(candidate1))

    # randomly choose which part of the individual we are going to mutate
    for individual in children :
        to_be_mutated = random.choice([k for k in individual])
        print("I am going to mutate part \"%s\"" % to_be_mutated)

        # different cases
        if to_be_mutated == "SC" :
            # pick another scale for the company, different from the current one
            sc_choices = [sc for sc in boundaries["SC"] if sc != individual["SC"] and sc in preferences["Sc"]]
            individual["SC"] = random.choice(sc_choices)

        elif to_be_mutated == "AIF" or to_be_mutated == "Nl" :
            # modify the quantity in (0,1) with a small Gaussian mutation
            individual[to_be_mutated] += random.gauss(0, 0.1)

        elif to_be_mutated == "EQ" :
            # this is easy, perform a random number of bit flips; low (high probability) or high (low probability)
            number_of_bit_flips = min(random.randint(1, len(individual["EQ"])) for i in range(0, len(individual["EQ"])))
            # choose several equipments (with replacement)
            indexes = random.sample(range(0, len(individual["EQ"])), number_of_bit_flips)

            for index in indexes :
                if individual["EQ"][index] == 0 :
                    individual["EQ"][index] = 1
                else :
                    individual["EQ"][index] = 0

        elif to_be_mutated == "F" :
            # perform a random number of value modifications; low (high probability) or high (low probability)
            number_of_modifications = min(random.randint(1, len(individual["F"])) for i in range(0, len(individual["F"])))
            # choose several types of feed (with replacement)
            print("Number of modifications: %d" % number_of_modifications)
            indexes = random.sample(range(0, len(individual["F"])), number_of_modifications)

            # small Gaussian mutation on each quantity
            for index in indexes :
                individual["F"][index] += random.gauss(0, 0.1)
        
        
        elif to_be_mutated == "C" :
            # pick another country to implement the company, different from the current one
            c_choices = [c for c in boundaries["C"] if c != individual["C"] and c in preferences["Sc"]] 
            individual["C"] = random.choice(c_choices)
                        
        elif to_be_mutated == "I" :
            start = True
            # perform a random number of bit flips; low (high probability) or high (low probability)
            while len(individual["I"]) == 0 or start == True :
                number_of_bit_flips = min(random.randint(1, len(individual["I"])) for i in range(0, len(individual["I"])))
                indexes = random.sample(range(0, len(individual["I"])), number_of_bit_flips)
                start = False
            
            individual_2 = list()
            for i in range(0, boundaries["I"]) :
                individual_2.append(individual["I"][i])
            for index in indexes :
                if individual["I"][index] == 0 and preferences["I"] != 0:
                    individual["I"][index] = 1
                else :
                    individual["I"][index] = 0
            if any(a != 0 for a in individual["I"]) == False:
                individual["I"].clear()
                for i in range(0, boundaries["I"]) :
                    individual["I"].append(individual_2[i])
               
                
        
        elif to_be_mutated == "r" :
            individual["r"] += random.gauss(0, 0.1)
        
        elif to_be_mutated == "theta" :
            individual["theta"] += random.gauss(0, 0.1)
        

    # after mutation or cross-over, check that the individual is still valid
    # in our case, we just need to normalize the amounts of each type of feed, and check that the
    # quantities in (0,1) are still in (0,1)
    for individual in children :

        # every element of "F" should also be between 0 and 1
        for i in range(0, boundaries["F"]) :
            if individual["F"][i] > 1.0 :
                individual["F"][i] = 1.0
            elif individual["F"][i] < 0.0 :
                individual["F"][i] = 0.0

        # sum of elements of "F" should be 1.0
        denominator = sum(individual["F"])

        for i in range(0, boundaries["F"]) :
            individual["F"][i] /= denominator

        for q in ["AIF", "Nl"] :
            if individual[q] > 1.0 :
                individual[q] = 1.0
            elif individual[q] < 0.0 :
                individual[q] = 0.0

    return children

def observer(population, num_generations, num_evaluations, args) :

    print("Generation %d (%d evaluations)" % (num_generations, num_evaluations))

    return



def variables_list( boundaries, json_preferences):

       
    preferences = dict()
    preferences["fd"] = list()
    for f in range(0, boundaries["F"]) :
        preferences["fd"].append(json_preferences["feed"][f])
    
    preferences["I"] = list()
    for i in range(0, boundaries["I"]) :
        preferences["I"].append(json_preferences["insect"][i])
    
    preferences["C"] = list()
    for c in range(0, len(json_preferences["region"])) :
        preferences["C"].append(json_preferences["region"][c])
    
    preferences["Sc"] = list()
    for sc in range(0, len(json_preferences["scaling"])):
        preferences["Sc"].append(json_preferences["scaling"][sc])
                
    preferences["Obj"] = list()
    for o in range(0, len(json_preferences["objectives"])):
        preferences["Obj"].append(json_preferences["objectives"][o])
    
      
    return preferences

def main() :
    
    # a few hard-coded parameters
    random_seed = 42

    # TODO also, we should do things properly and create a log file

    # load information on the problem
    json_instance = load_instance('../data/data_project_with_preferences.json') 
    json_preferences = load_instance('../data/preferences.json') 
    
    # boundaries for all the values included in the individual
    boundaries = dict()
    boundaries["SC"] = [1, 2, 3, 4] # minimum and maximum
    boundaries["EQ"] = 5 # number of different types of equipments
    boundaries["F"] = 5 # types of different feeds
    boundaries["C"] = [1, 2, 3, 4]  # enumeration of diferent cities
    boundaries["I"] = 4 # types of different insects 

    # boundaries for AIF and Nl, depending on SC
    boundaries["AIF"] = dict()
    boundaries["AIF"][1] = [25000, 75000]
    boundaries["AIF"][2] = [75000, 125000]
    boundaries["AIF"][3] = [125000, 175000]
    boundaries["AIF"][4] = [175000, 250000]    
    
    boundaries["Nl"] = dict()
    boundaries["Nl"][1] = [25, 75]
    boundaries["Nl"][2] = [75, 125]
    boundaries["Nl"][3] = [125, 175]
    boundaries["Nl"][4] = [175, 250]
    
    # boundaries for RW depending on C (country)
    boundaries["RW"] = dict()
    boundaries["RW"][0] = [1508, 2659]
    boundaries["RW"][1] = [1708, 2649]
    boundaries["RW"][2] = [1608, 2639]
    boundaries["RW"][3] = [1208, 2679]
  
    preferences = variables_list(boundaries, json_preferences)
    
    # initialize random number generator
    random_number_generator = random.Random()
    random_number_generator.seed(random_seed)

    # create instance of NSGA2
    nsga2 = inspyred.ec.emo.NSGA2(random_number_generator)
    nsga2.observer = observer
    nsga2.terminator = inspyred.ec.terminators.evaluation_termination # stop after a certain number of evaluations
    nsga2.variator = [variator] # types of evolutionary operators to be used
    
    final_pareto_front = nsga2.evolve(
                            generator = generator,
                            evaluator = evaluator,
                            pop_size = 1000,
                            num_selected = 2000,
                            max_evaluations = 20000,
                            maximize = True, # TODO currently, it's trying to maximize EVERYTHING, so we need to 
                                             # have the fitness function output values that are better than higher
                                             # for example, use 1.0/value if the initial idea was to minimize

                            # all arguments specified below, THAT ARE NOT part of the "evolve" method, will be automatically placed in "args"
                            # "args" is a dictionary that is passed to all functions
                            boundaries = boundaries,
                            preferences = preferences,
                            json_instance = json_instance,
    )

    # save the final Pareto front in a .csv file
    # prepare dictionary that will be later converted to .csv file using Pandas library
    df_dictionary = { "SC": [], "Nl": [], "AIF": [], "RW": [], "C": [], "r": [], "theta": [], "Economic_Impact": [], "Environmental_Impact": [],"Social_Impact": []} 

    for e in range(0, boundaries["EQ"]) :
        df_dictionary["EQ" + str(e)] = []
    for f in range(0, boundaries["F"]) :
        df_dictionary["F" + str(f)] = []
    for g in range(0, boundaries["I"]) :
        df_dictionary["I" + str(g)] = []

    # TODO change names of the fitnesses to their appropriate correspondence (e.g. "Profit", "Social Impact", "Environmental Impact")
    #df_dictionary["Economic_Impact"] = []
    #df_dictionary["Environmental_Impact"] = []
    #df_dictionary["Social_Impact"] = []

    # go over the list of individuals in the Pareto front and store them in the dictionary 
    # after converting them from the internal 'genome' representation to actual values
    L = 0
    for individual in final_pareto_front :
        #genome = individual.genome # uncomment this line and comment the two lines below to have the individuals saved with their internal representation
        SC, Nl, AIF, F, EQ, RW, I, C, r, theta  = convert_individual_to_values(individual.candidate, boundaries)
 
        val_1= individual.fitness[0]
        val_2= individual.fitness[1]
        val_3= individual.fitness[2]
        
        genome = { "SC": SC, "Nl": Nl, "AIF": AIF, "F": F, "EQ": EQ, "RW": RW, "C": C, "I": I, "r": r, "theta": theta, "Economic_Impact": val_1, "Environmental_Impact": 1/val_2, "Social_Impact": val_3}

        if L %200 == 0:
            for k in genome:
                # manage parts of the genome who are lists
                if isinstance(genome[k], list) :
                    for i in range(0, len(genome[k])) :
                        df_dictionary[k + str(i)].append(genome[k][i])
                else:
                    df_dictionary[k].append(genome[k])             

            df = pd.DataFrame.from_dict(df_dictionary)
            df.to_csv("pareto-front_2.csv", index=False)
        L += 1   
        
       
    return

# calls the 'main' function when the script is called
if __name__ == "__main__" :
    sys.exit( main() )
