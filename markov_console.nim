import random, strutils

type
  MarkovChain = ref object
    transitions: seq[seq[float]]  # Probabilities of moving from one state to another
    states: seq[string]           # Possible states (characters)

proc initMarkovChain(states: seq[string], probabilities: seq[seq[float]]): MarkovChain =
  result = MarkovChain()
  result.states = states
  result.transitions = probabilities

proc nextState(mc: MarkovChain, currentState: int): int =
  let r = rand(1.0)  # Generate a random float between 0 and 1
  var cumProb = 0.0
  for i, prob in mc.transitions[currentState]:
    cumProb += prob
    if r < cumProb:
      return i
  return currentState  # In case something goes wrong, stay in current state

proc displayMarkovChain(mc: MarkovChain, iterations: int) =
  var currentState = 0
  for i in 0..<iterations:
    let next = nextState(mc, currentState)
    stdout.write(mc.states[next])  # Print the next state (character)
    if (i+1) mod 50 == 0:          # Every 50 characters, add a newline for visual structure
      stdout.write("\n")
    currentState = next

# Example Markov chain setup
let
  states = @["*", "#", ".", " "]
  # Transition matrix for moving from one character to another
  transitions = @[
    @[0.1, 0.6, 0.2, 0.1],  # Probabilities of transitioning from state "*"
    @[0.5, 0.2, 0.1, 0.2],  # Probabilities of transitioning from state "#"
    @[0.3, 0.1, 0.4, 0.2],  # Probabilities of transitioning from state "."
    @[0.2, 0.2, 0.3, 0.3]   # Probabilities of transitioning from state " "
  ]

# Create the Markov chain
let mc = initMarkovChain(states, transitions)

# Run the Markov chain for 500 iterations and display output in the console
displayMarkovChain(mc, 500)
