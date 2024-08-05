Creating an inter-process communication (IPC) system using POSIX pipes in Nim involves several steps. POSIX pipes allow two processes to communicate through a unidirectional channel. Below is an example that demonstrates how to set up a simple IPC system using pipes in Nim.

### Step 1: Create the Pipe

First, we need to create a pipe using the `pipe` system call. This will create two file descriptors: one for reading and one for writing.

### Step 2: Fork the Process

Next, we will fork the process using the `fork` system call. The parent process will write to the pipe, and the child process will read from the pipe.

### Step 3: Implement the Communication

The parent process will write a message to the pipe, and the child process will read the message from the pipe.

Here is the complete example:

```nim
import posix

const MSG_SIZE = 256

proc main() =
  var pipefd: array[2, cint]
  var msg: array[MSG_SIZE, char]

  # Create the pipe
  if pipe(pipefd) == -1:
    echo "Pipe creation failed"
    return

  # Fork the process
  let pid = fork()
  if pid == -1:
    echo "Fork failed"
    return

  if pid > 0:
    # Parent process
    close(pipefd[0])  # Close the read end of the pipe

    const message = "Hello from parent process"
    echo "Parent process writing to pipe: ", message
    discard write(pipefd[1], message.cstring, message.len + 1)

    close(pipefd[1])  # Close the write end of the pipe
  else:
    # Child process
    close(pipefd[1])  # Close the write end of the pipe

    let bytesRead = read(pipefd[0], addr msg[0], MSG_SIZE)
    if bytesRead > 0:
      echo "Child process received message: ", msg.toString()

    close(pipefd[0])  # Close the read end of the pipe

main()
```

### Explanation

1. **Pipe Creation**: The `pipe` system call creates a pipe and returns two file descriptors, `pipefd[0]` for reading and `pipefd[1]` for writing.

2. **Forking the Process**: The `fork` system call creates a child process. The return value of `fork` is 0 for the child process and the PID of the child process for the parent process.

3. **Parent Process**:
   - Close the read end of the pipe (`pipefd[0]`).
   - Write a message to the pipe using the `write` system call.
   - Close the write end of the pipe (`pipefd[1]`).

4. **Child Process**:
   - Close the write end of the pipe (`pipefd[1]`).
   - Read the message from the pipe using the `read` system call.
   - Close the read end of the pipe (`pipefd[0]`).

### Running the Example

To run this example, save the code to a file (e.g., `ipc_pipe.nim`) and compile it using the Nim compiler:

```sh
nim c -r ipc_pipe.nim
```

This will compile and run the program, demonstrating inter-process communication using POSIX pipes in Nim.
