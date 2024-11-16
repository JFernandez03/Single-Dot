from SingleDotProblem import State, Problem
from problemGraphics import pacmanGraphic
import random

p = Problem('singleDotSmall.txt')  # Use 'singleDotMedium.txt' if you want a bigger maze

# Initialize V as an empty dictionary
V = {}

# Set a value for gamma (discount factor)
gamma = 0.9  # or another value based on your requirements

# Set epsilon for exploration
epsilon = 0.1

# Initialize the pacman graphic
pac = pacmanGraphic(1300, 700)
pac.setup(p)

# Start the main loop for value iteration
for episode in range(100):  # Loop for 100 episodes
    print(f"Episode {episode + 1}")

    # Set the current state as the problem start state
    currentState = p.getStartState()

    # Initialize visited states to prevent infinite loops
    visited_states = set()
    max_iterations = 1000
    iterations = 0

    # Start the infinite loop (until a terminal state is reached)
    while True:
        # Check if the current state is a terminal state
        if p.isTerminal(currentState):
            V[currentState] = p.reward(currentState)
            break  # Exit the loop if we reached a terminal state

        # If currentState is not in V, set its value to 0
        if currentState not in V:
            V[currentState] = 0

        # Get the valid neighbors (state-action pairs)
        neighbors = p.transition(currentState)

        # Prevent infinite loops by tracking visited states
        if currentState in visited_states:
            print("Stuck in a loop, exiting...")
            break

        visited_states.add(currentState)

        if not neighbors:
            print("No valid moves, exiting...")
            break  # If there are no valid moves, exit the loop

        # Compute the maximum reward for all neighbors
        maxV = -99999  # Initialize to a very low value
        bestState = None  # Best state for the maximum reward
        for s, a in neighbors:
            if s in V:
                v = V[s]  # Use the value of the neighbor if it's already in V
            else:
                v = 0  # Otherwise, assume its value is 0
            if v > maxV:
                maxV = v
                bestState = s

        # Update the value for the current state using the Bellman equation
        V[currentState] = p.reward(currentState) + gamma * maxV

        # Exploration or exploitation step (epsilon-greedy)
        r = random.random()
        if r < epsilon:  # Exploration
            n, a = random.choice(neighbors)
        else:  # Exploitation
            n = bestState

        # Check if the next state is the same as the current one (avoid infinite loops)
        if n == currentState:
            print("Next state is the same as the current state, breaking...")
            break

        # Check if the next state leads to a wall (invalid state)
        # Assuming the 'transition' function checks for valid moves
        if p.isWall(n.agentPos):  # Assuming there's a function `isWall` in your Problem class
            print(f"Invalid move: Wall detected at {n.agentPos}, trying again...")
            continue  # Skip this iteration and try another action

        currentState = n  # Move to the next state
        iterations += 1
        if iterations > max_iterations:
            print("Too many iterations, breaking the loop...")
            break  # Break the loop if the number of iterations is exceeded

# Now extract the policy from the value function (V)
policy = {}

# Loop through all states in V and extract the best action for each state
for s in V:
    if p.isTerminal(s):
        policy[s] = None  # No action needed for terminal states
        continue

    neighbors = p.transition(s)
    maxV = -9999  # Initialize maxV to a low value
    bestAction = None

    for n, a in neighbors:
        if n in V:
            v = V[n]  # Use the value of the neighbor if it's in V
        else:
            v = 0  # Assume the value is 0 if not found in V

        if v > maxV:
            maxV = v
            bestAction = a

    # Save the best action for the current state
    policy[s] = bestAction

# Visualize the learned policy
for k in policy:
    if policy[k] == 'L':
        s = '\u2190'  # Left arrow
    elif policy[k] == 'R':
        s = '\u2192'  # Right arrow
    elif policy[k] == 'U':
        s = '\u2191'  # Up arrow
    elif policy[k] == 'D':
        s = '\u2193'  # Down arrow
    pac.addText(k.agentPos[0] + 0.5, k.agentPos[1] + 0.5, s, fontSize=20)

# Run the learned policy to show Pacman moving through the maze
currentState = p.getStartState()
count = 0
while currentState:
    action = policy[currentState]
    if action is None:
        break  # No action means terminal state reached

    count += 1
    dx, dy = p.potential_moves[action]
    agentPos = (currentState.agentPos[0] + dx, currentState.agentPos[1] + dy)

    # Check if the next position is a wall
    if p.isWall(agentPos):
        print(f"Attempted to move into a wall at {agentPos}. Skipping move.")
        continue  # Skip this iteration if the move is invalid (into a wall)

    if agentPos in p.dots:
        index = p.dots.index(agentPos)
        pac.remove_dot(index)

    currentState = State(agentPos)  # Create a new state with the updated position
    pac.move_pacman(dx, dy)

print(f"Plan length = {count}")
