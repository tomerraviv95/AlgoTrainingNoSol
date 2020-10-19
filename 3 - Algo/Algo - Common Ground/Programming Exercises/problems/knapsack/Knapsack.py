"""
problem formalization:

You are going on a trip and arrange your knapsack.
You are willing to take with you N objects. each object is assigned 'Value', integer representing how
important it is to you.
In addition, each object has 'Weight' in kg. the problem is that your knapsack has limited capacity, so it
can contain at most C kg of equipment.

Create an EFFICIENT algorithm to choose which of the objects you take with you, maximizing the total value of objects,
while maintaining the capacity limit of weights.

-----------------------------------------------------------------------------------------------------------------------

Example:

input: Value: [2, 5, 8, 11], Weight: [4, 9, 2, 10], C: 14
output: 19

-----------------------------------------------------------------------------------------------------------------------

Limitations:

time - 0.1 seconds

-----------------------------------------------------------------------------------------------------------------------

Testing:

After implementing your solution, test it with our given input by 'CheckSolution' file.
You have a total of 6 test:
- tests 1-4 are visible to you, and you can access the input using 'get_input' method from utils.Test.
- tests 5-6 are not visible to you, and need to pass them without knowing the input.
It is assured to you that all input is legal and fits the solution signature.

-----------------------------------------------------------------------------------------------------------------------

Documentation:

After passing all tests, write a doc in Confluence describing your solution.
In the doc, analyze the runtime of the algorithm you used.

"""

from typing import List


def knapsack_solution(v: List[int], w: List[float], c: float) -> int:
    """ Decides which items to take, by maximizing the total value of them

    :param v: list of size N containing the values of the objects (all integers)
    :param w: list of size N containing the weights of the objects (may not be integers)
    :param c: the backpack weight capacity
    :return: the total value possible under the limitations
    """
    pass