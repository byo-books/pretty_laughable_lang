# The Pretty Laughable Programming Language

An educational C-like toy programming language that compiles to x64 binary.

The [compiler](pl_comp.py) is a self-contained Python program that weighs about 1000 LoC.
It's part of an online [tutorial](https://build-your-own.org/b2a/p0_intro.html) on compilers and interpreters.

## Introduction

The hello world looks like this:

```clojure
; the write() syscall:
; ssize_t write(int fd, const void *buf, size_t count);
(syscall 1 1 "Hello world!\n" 13)
0
```

Compile and run the program:
```sh
$ ./pl_comp.py ./samples/hello.txt -o ./hello
$ ./hello
Hello world!
```

The output is a tiny freestanding x64 Linux ELF binary.
```sh
$ file hello
hello: ELF 64-bit LSB executable, x86-64, version 1 (SYSV), statically linked, no section header
$ wc -c hello
288 hello
```

## The Language

The syntax is just S-expression, parsing strings is too boring for me.

The semantics are C-like. The only data types are integers and pointers. This should be enough to write any program in.

### 01. Pointers

The `peek` command reads data from a pointer and the `poke` command writes to a pointer.

```clojure
; copy data byte by byte
(def (memcpy void) ((dst ptr byte) (src ptr byte) (n int)) (do
    (loop n (do
        (poke dst (peek src))
        (set dst (+ 1 dst))
        (set src (+ 1 src))
        (set n (- n 1))
    ))
))
```

### 02. Control Flows

List of control flow structures:

```clojure
(? cond yes no)
(if cond (then yes blah blah) (else no no no))
(do a b c...)
(loop cond body)
(break)
(continue)
(call f a b c...)
(return val)
```

Some examples:

```clojure
(def (fib int) ((n int))
    (if (le n 0) (then 0) (else (+ n (call fib (- n 1))))))
```

```clojure
(def (fib int) ((n int)) (do
    (var r 0)
    (loop (gt n 0) (do
        (set r (+ r n))
        (set n (- n 1))
    ))
    (return r)
))
```

### 03. Data Types

The only data types are:

- `byte`: unsigned 8-bit integer.
- `int`:  signed 64-bit integer.
- `ptr elem_type`: pointer to `elem_type`.

Variable types are automatically inferred:

```clojure
(var a 123)         ; int
(var b 45u8)        ; byte
(var p (ptr int))   ; a null pointer to int
(var s "asdf")      ; ptr byte
```

The type of the function return value and the argument must be specified explicitly:

```clojure
(def (memcpy void) ((dst ptr byte) (src ptr byte) (n int)) (do
    ; ...
))
```

`int` can be cast to any pointer types and vice versa.

```clojure
(var i 0x1234)                  ; int
(var p (cast (ptr int) i))      ; ptr int
(var a (cast (int) (+ 1 p)))    ; int
```

### 04. Memory Management

Memory management is very simple at this point, because it doesn't exist at all.

However, the language doesn't prevent you from building your own memory management routines. This usually starts with the `mmap` syscall.

```clojure
(var heap (ptr byte))

; a fake malloc
(def (malloc ptr byte) ((n int)) (do
    (if (not heap) (do
        ; create the heap via mmap()
        (var heapsz 1048576)    ; 1M
        (var prot 3)            ; PROT_READ|PROT_WRITE
        (var flags 0x22)        ; MAP_PRIVATE|MAP_ANONYMOUS
        (var fd -1)
        (var offset 0)
        (var r (syscall 9 0 heapsz prot flags fd offset))
        (set heap (cast (ptr byte) r))
    ))
    ; just move the heap pointer forward
    (var r heap)
    (set heap (+ n heap))
    (return r)
))

; TODO: figure out how to recycle the memory
(def (free void) ((p ptr byte)) (do))
```

### 05. The stdlib

The Pretty Laughable Language comes with the world's smallest standard library &mdash; no standard library &mdash; not even a builtin `print` function.

But with the ability to make arbitrary syscalls and peek-poke the memory, you can build your own stdlibs. Let's add the `print` function:

```clojure
(def (strlen int) ((s ptr byte)) (do
    (var start s)
    (loop (peek s) (set s (+ 1 s)))
    (return (- s start))
))

(def (print void) ((s ptr byte)) (do
    (syscall 1 1 s (call strlen s))
))

(call print "Yes!\n")
0
```

[Here](samples/malloc_and_strings.txt) is a more sophisticated program you can play with.

## Roadmaps

Language features:

- [x] int, byte
- [x] pointer
- [x] syscall
- [x] if-then-else, loop
- [x] function
- [x] nested function, nonlocal variable
- [ ] array
- [ ] struct, class
- [ ] function pointer

Explorations:

- [ ] module or `include` directive
- [ ] macro?
- [ ] alternative syntax?
- [ ] Windows
- [ ] ARM64
- [ ] WASM

Optimizations:

- [ ] register allocation
- [ ] constants
- [ ] tail call

## The Design

### 01. The Goal

TBA

### 02. The IR (Intermediate Representation)

TBA

### 03. Machine Code Generation

TBA

## The Implementation

To be added.

But you can learn how to do it by reading the source code.

Or you might like the online [tutorial](https://build-your-own.org/b2a/p0_intro.html) on building your own compiler,
which is part of the WIP book "[5 Cool Coding Challenges from Beginner to Advanced](https://build-your-own.org/b2a/)".
