Creating an inter-process communication (IPC) system using POSIX pipes in Nim involves setting up a parent process that creates a pipe and a child process that reads from or writes to that pipe. Below is an example demonstrating how to achieve this using Nim's `osproc` module.

### Step 1: Parent Process

The parent process will create a pipe, fork a child process, and then write data to the pipe.

#### parent.nim

```nim
import os, osproc, strutils

const
  READ_END = 0
  WRITE_END = 1

proc createPipe(): (int, int) =
  var fds: array[2, int]
  if pipe(fds) == -1:
    quit("Failed to create pipe")
  (fds[READ_END], fds[WRITE_END])

proc parent() =
  let (readFd, writeFd) = createPipe()
  let childProcess = startProcess(command = "child", args = [readFd.toString])

  defer:
    close(writeFd)
    close(childProcess)

  let message = "Hello from parent"
  let bytesWritten = write(writeFd, cast[pointer](message.cstring), message.len)
  if bytesWritten == -1:
    quit("Failed to write to pipe")
  else:
    echo "Parent: Written ", bytesWritten, " bytes to pipe"

parent()
```

### Step 2: Child Process

The child process will read data from the pipe and print it to the console.

#### child.nim

```nim
import os, strutils

const
  READ_END = 0

proc child(pipeFd: int) =
  var buffer = newString(1024)
  let bytesRead = read(pipeFd, cast[pointer](buffer.cstring), buffer.len)
  if bytesRead == -1:
    quit("Failed to read from pipe")
  else:
    buffer.setLen(bytesRead)
    echo "Child: Read ", bytesRead, " bytes from pipe: ", buffer

if paramCount() == 1:
  let pipeFd = paramStr(1).parseInt()
  child(pipeFd)
else:
  quit("Usage: child <pipe_fd>")
```

### Step 3: Compile and Run

To compile and run the example, follow these steps:

1. **Compile the Parent Process**:

```bash
nim c -o:parent parent.nim
```

2. **Compile the Child Process**:

```bash
nim c -o:child child.nim
```

3. **Run the Parent Process**:

```bash
./parent
```

### Explanation

1. **Creating the Pipe**:
   - The `createPipe` procedure creates a pipe using the `pipe` system call. It returns a tuple containing the read and write file descriptors.

2. **Parent Process**:
   - The `parent` procedure creates a pipe and then starts the child process, passing the read file descriptor as an argument.
   - The parent process writes a message to the pipe and then closes the write end of the pipe.

3. **Child Process**:
   - The `child` procedure reads from the pipe using the file descriptor passed as an argument.
   - It prints the number of bytes read and the message to the console.

4. **Running the Programs**:
   - Compile the parent and child programs separately.
   - Run the parent program, which will start the child process and handle the pipe communication.

This setup demonstrates basic IPC using POSIX pipes in Nim, with a parent process sending a message to a child process. The child process reads the message from the pipe and prints it.
