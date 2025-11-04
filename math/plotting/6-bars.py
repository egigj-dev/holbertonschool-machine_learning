#!/usr/bin/env python3
"""Script that plots stack bar graph"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """Function that plots a stack bar graph"""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))
    persons = ['Farrah', 'Fred', 'Felicia']
    x = np.arange(len(persons))
    apples = fruit[0]
    bananas = fruit[1]
    oranges = fruit[2]
    peaches = fruit[3]

    # Stack bars
    plt.bar(x, apples, width=0.5, color='red', label='apples')
    plt.bar(x, 
            bananas, 
            width=0.5, 
            bottom=apples, 
            color='yellow', 
            label='bananas')
    plt.bar(x, 
            oranges, 
            width=0.5, 
            bottom=apples + bananas, 
            color='#ff8000', 
            label='oranges')
    plt.bar(x, 
            peaches, 
            width=0.5, 
            bottom=apples + bananas + oranges, 
            color='#ffe5b4', 
            label='peaches')

    # Labels and title
    plt.ylabel("Quantity of Fruit")
    plt.title("Number of Fruit per Person")

    # X-axis ticks
    plt.xticks(x, persons)

    # Y-axis ticks 0 → 80 by 10
    plt.yticks(np.arange(0, 81, 10))
    plt.ylim(0, 80)

    # Legend
    plt.legend()
    plt.show()
