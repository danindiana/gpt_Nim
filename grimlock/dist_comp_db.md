Setting up a distributed compute database using Nim and Arraymancer involves several steps, including setting up key-value stores on each node, establishing communication between nodes, and distributing computations. Below is a detailed guide to achieve this:

### Step 1: Setting Up Key-Value Stores on Each Node

First, we need to create a simple in-memory key-value store for each node. Each node will store its own data and communicate with other nodes to perform distributed computations.

#### Key-Value Store (Nim)
```nim
import tables

type
  KVStore = ref object
    store: Table[string, string]

proc newKVStore(): KVStore =
  new(result)
  result.store = initTable[string, string]()

proc get(self: KVStore, key: string): string =
  return self.store.getOrDefault(key, "")

proc set(self: KVStore, key: string, value: string) =
  self.store[key] = value

var kvStore = newKVStore()
kvStore.set("example_key", "example_value")
echo kvStore.get("example_key")
```

### Step 2: Establishing Communication Between Nodes

We'll use sockets for communication between nodes. Each node will act as both a server and a client.

#### Server (Nim)
```nim
import socketserver

type
  Handler = ref object of RequestHandler
    data: string

method handle(self: Handler) =
  self.data = self.request.recv(1024).strip()
  echo "Received: ", self.data
  self.request.send("ACK".cstring)

var server = new ThreadedTCPServer()
server.serve(Port(9998))
```

#### Client (Nim)
```nim
import socket

proc connect_to_server(host: string, port: int, data: string): bool =
  var sock = newSocket()
  try:
    sock.connect(host, Port(port))
    sock.send(data)
    let response = sock.recv(1024)
    echo "Server response: ", response
    return true
  except:
    echo "Connection failed"
    return false
  finally:
    sock.close()

discard connect_to_server("localhost", 9998, "Hello, Server!")
```

### Step 3: Distributed Computation with Arraymancer

Each node will compute gradients locally and store the results in the key-value store. The central node will aggregate the results from all nodes.

#### Node Computation (Nim)
```nim
import arraymancer
import json
import socketserver
import socket
import tables

proc compute_gradients(X: Tensor[float32], y: Tensor[float32], net: Context[float32]): Tensor[float32] =
  let y_pred = net.forward(X)
  let loss = net.mse_loss().forward(y_pred, y)
  net.zero_grad()
  loss.backward()
  return net.parameters.mapIt(it.grad)

proc connect_to_server(host: string, port: int, data: string): bool =
  var sock = newSocket()
  try:
    sock.connect(host, Port(port))
    sock.send(data)
    let response = sock.recv(1024)
    echo "Server response: ", response
    return true
  except:
    echo "Connection failed"
    return false
  finally:
    sock.close()

let X = @[@[0.0, 0.0], @[0.0, 1.0], @[1.0, 0.0], @[1.0, 1.0]].toTensor()
let y = @[@[0.0], @[1.0], @[1.0], @[0.0]].toTensor()

let ctx = newContext[float32]()
let net = ctx.sequential(
  ctx.linear(2, 2),
  ctx.relu(),
  ctx.linear(2, 1)
)

let gradients = compute_gradients(X, y, net)
let gradients_json = gradients.toJson()

# Store gradients in key-value store
var kvStore = newKVStore()
kvStore.set("gradients", gradients_json)

# Send gradients to central node
discard connect_to_server("localhost", 9998, gradients_json)
```

#### Server (Nim)
```nim
import socketserver
import tables

type
  Handler = ref object of RequestHandler
    data: string

method handle(self: Handler) =
  self.data = self.request.recv(1024).strip()
  echo "Received: ", self.data
  self.request.send("ACK".cstring)

var server = new ThreadedTCPServer()
server.serve(Port(9998))
```

#### Aggregation (Nim)
```nim
import arraymancer
import json
import tables

proc aggregate_gradients(gradients_list: seq[Tensor[float32]]): Tensor[float32] =
  var aggregated_gradients = gradients_list[0]
  for i in 1..<gradients_list.len:
    aggregated_gradients += gradients_list[i]
  return aggregated_gradients / gradients_list.len.float32

var gradients_list: seq[Tensor[float32]] = @[]

# Receive gradients from nodes
# This part would be implemented in the server handler
# For simplicity, we simulate it here
let gradients_json = """[{"data": [0.1, 0.2, 0.3, 0.4], "shape": [2, 2]}, {"data": [0.5, 0.6, 0.7, 0.8], "shape": [2, 2]}]"""
let gradients = json.parseTensor[float32](gradients_json)
gradients_list.add(gradients)

let aggregated_gradients = aggregate_gradients(gradients_list)
echo "Aggregated Gradients: ", aggregated_gradients
```

### Putting It All Together

To implement a distributed compute database, you would run the key-value store and server on each node and have the client send gradients to the server for aggregation. The central node would then aggregate the gradients and update the model parameters.

This example is simplified and does not cover all aspects of a distributed compute database, such as fault tolerance, security, and efficient data distribution. However, it provides a basic framework for understanding how communication, computation, and data storage can be integrated in a distributed system using Nim and Arraymancer.

For a more robust implementation, you would need to consider additional libraries and frameworks that support distributed computing and databases, such as Apache Spark, Apache Cassandra, or Redis, and potentially integrate them with Nim using bindings or custom implementations.
