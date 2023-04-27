#!/usr/bin/env python3


import struct


# TODO: function pointers
# TODO: pointer to variables
# TODO: binop8, unop8
# TODO: array
# TODO: class or struct
# TODO: node comment
# TODO: module or include directive

# TODO: register allocation
# TODO: constant prop
# TODO: tail call


def skip_space(s, idx):
    while True:
        save = idx
        # spaces
        while idx < len(s) and s[idx].isspace():
            idx += 1
        # line comment
        if idx < len(s) and s[idx] == ';':
            idx += 1
            while idx < len(s) and s[idx] != '\n':
                idx += 1
        if idx == save:
            break
    return idx


def parse_expr(s, idx):
    idx = skip_space(s, idx)
    if s[idx] == '(':
        idx += 1
        l = []
        while True:
            idx = skip_space(s, idx)
            if idx >= len(s):
                raise Exception('unbalanced parenthesis')
            if s[idx] == ')':
                idx += 1
                break
            idx, v = parse_expr(s, idx)
            l.append(v)
        return idx, l
    elif s[idx] == ')':
        raise Exception('bad parenthesis')
    elif s[idx] == '"' or s[idx] == "'":
        # string or u8
        return parse_quotes(s, idx)
    else:
        # constant or name
        start = idx
        while idx < len(s) and (not s[idx].isspace()) and s[idx] not in '()':
            idx += 1
        if start == idx:
            raise Exception('empty program')
        return idx, parse_value(s[start:idx])


def parse_quotes(s, idx):
    term = s[idx]
    end = idx + 1
    while end < len(s):
        if s[end] == term:
            break
        if s[end] == '\\':
            end += 1
        end += 1
    if end < len(s) and s[end] == term:
        # TODO: actually implement this
        import json
        v = json.loads('"' + s[idx+1:end] + '"')
        if term == '"':
            v = ['str', v]
        else:
            if len(v) != 1:
                raise Exception('bad char')
            v = ord(v)
            if not (0 <= v < 256):
                raise ValueError('bad integer range')
            v = ['val8', v]
        return end + 1, v


# a single constant, or a name
def parse_value(s):
    # int
    try:
        v = try_int(s)
    except ValueError:
        pass
    else:
        if not (-(1 << 63) <= v < (1 << 63)):
            raise ValueError('bad integer range')
        return ['val', v]

    # u8
    if s.endswith('u8'):
        try:
            v = try_int(s[:-2])
        except ValueError:
            pass
        else:
            if not (0 <= v < 256):
                raise ValueError('bad integer range')
            return ['val8', v]

    # other
    if s[0].isdigit():
        raise ValueError('bad name')
    return s


def try_int(s):
    base = 10
    if s[:2].lower() == '0x':
        base = 16
    # TODO: other bases
    return int(s, base)


def pl_parse(s):
    idx, node = parse_expr(s, 0)
    idx = skip_space(s, idx)
    assert idx == len(s)
    return node


def pl_parse_main(s):
    return pl_parse('(def (main int) () (do ' + s + '))')


class Func:
    def __init__(self, prev):
        self.prev = prev
        self.root = prev.root if prev else self
        self.level = (prev.level + 1) if prev else 0
        self.scope = Scope(None)
        self.nvar = 0
        self.code = []
        self.stack = 0
        self.labels = dict()
        self.rtype = None
        # root
        self.funcs = []

    def scope_enter(self):
        self.scope = Scope(self.scope)
        self.scope.save = self.stack

    def scope_leave(self):
        self.stack = self.scope.save
        self.nvar -= self.scope.nlocal
        self.scope = self.scope.prev

    def add_var(self, name, tp):
        if name in self.scope.names:
            raise ValueError('duplicated name')
        self.scope.names[name] = (tp, self.nvar)
        self.scope.nlocal += 1
        self.nvar += 1

    def get_var(self, name):
        tp, var = scope_get_var(self.scope, name)
        if var >= 0:
            return (self.level, tp, var)
        if not self.prev:
            raise ValueError('undefined name')
        return self.prev.get_var(name)

    def tmp(self):
        dst = self.stack
        self.stack += 1
        return dst

    def new_label(self):
        l = len(self.labels)
        self.labels[l] = None
        return l

    def set_label(self, l):
        assert l in self.labels
        self.labels[l] = len(self.code)


class Scope:
    def __init__(self, prev):
        self.prev = prev
        self.nlocal = 0
        self.names = dict()
        self.loop_start = prev.loop_start if prev else -1
        self.loop_end = prev.loop_end if prev else -1


def scope_get_var(scope, name):
    while scope:
        if name in scope.names:
            return scope.names[name]
        scope = scope.prev
    return None, -1


def pl_comp_expr(fenv: Func, node, *, allow_var=False):
    if allow_var:
        assert fenv.stack == fenv.nvar
    save = fenv.stack

    tp, var = pl_comp_expr_tmp(fenv, node, allow_var=allow_var)
    assert var < fenv.stack

    if allow_var:
        fenv.stack = fenv.nvar
    else:
        fenv.stack = save
    assert var <= fenv.stack
    return tp, var


def pl_comp_getvar(fenv: Func, node):
    assert isinstance(node, str)
    flevel, tp, var = fenv.get_var(node)
    if flevel == fenv.level:
        # local var
        return tp, var
    else:
        # outer env
        dst = fenv.tmp()
        fenv.code.append(('get_env', flevel, var, dst))
        return tp, dst


def pl_comp_const(fenv: Func, node):
    _, kid = node
    assert isinstance(kid, (int, str))
    dst = fenv.tmp()
    fenv.code.append(('const', kid, dst))
    tp = dict(val='int', val8='byte', str='ptr byte')[node[0]]
    tp = tuple(tp.split())
    return tp, dst


def pl_comp_binop(fenv: Func, node):
    op, lhs, rhs = node

    save = fenv.stack
    t1, a1 = pl_comp_expr_tmp(fenv, lhs)
    t2, a2 = pl_comp_expr_tmp(fenv, rhs)
    fenv.stack = save

    # pointers
    if op == '+' and (t1[0], t2[0]) == ('int', 'ptr'):
        t1, a1, t2, a2 = t2, a2, t1, a1
    if op in '+-' and (t1[0], t2[0]) == ('ptr', 'int'):
        scale = 8
        if t1 == ('ptr', 'byte'):
            scale = 1
        if op == '-':
            scale = -scale
        fenv.code.append(('lea', a1, a2, scale, fenv.stack))
        return t1, fenv.tmp()

    # check types
    # TODO: allow different types
    cmp = {'eq', 'ge', 'gt', 'le', 'lt', 'ne', '-'}
    ints = (t1 == t2 and t1[0] in ('int', 'byte'))
    ptr_cmp = (t1 == t2 and t1[0] == 'ptr' and op in cmp)
    if not (ints or ptr_cmp):
        raise ValueError('bad binop types')
    rtype = t1
    if ptr_cmp and op == '-':
        rtype = ('int',)

    suffix = ''
    if t1 == t2 and t1 == ('byte',):
        suffix = '8'
    fenv.code.append(('binop' + suffix, op, a1, a2, fenv.stack))
    return rtype, fenv.tmp()
    # FIXME: boolean short circuit


def pl_comp_unop(fenv: Func, node):
    op, arg = node
    t1, a1 = pl_comp_expr(fenv, arg)

    suffix = ''
    rtype = t1
    if op == '-':
        if t1[0] not in ('int', 'byte'):
            raise ValueError('bad unop types')
        if t1 == ('byte',):
            suffix = '8'
    elif op == 'not':
        if t1[0] not in ('int', 'byte', 'ptr'):
            raise ValueError('bad unop types')
        rtype = 'int'   # boolean
    fenv.code.append(('unop' + suffix, op, a1, fenv.stack))
    return rtype, fenv.tmp()


def pl_comp_expr_tmp(fenv: Func, node, *, allow_var=False):
    # read a variable
    if not isinstance(node, list):
        return pl_comp_getvar(fenv, node)

    # anything else
    if len(node) == 0:
        raise ValueError('empty list')

    # constant
    if len(node) == 2 and node[0] in ('val', 'val8', 'str'):
        return pl_comp_const(fenv, node)
    # binary operators
    binops = {
        '%', '*', '+', '-', '/',
        'and', 'or',
        'eq', 'ge', 'gt', 'le', 'lt', 'ne',
    }
    if len(node) == 3 and node[0] in binops:
        return pl_comp_binop(fenv, node)
    # unary operators
    if len(node) == 2 and node[0] in {'-', 'not'}:
        return pl_comp_unop(fenv, node)
    # conditional
    if len(node) in (3, 4) and node[0] in ('?', 'if'):
        return pl_comp_cond(fenv, node)
    # new scope
    if node[0] in ('do', 'then', 'else'):
        return pl_comp_scope(fenv, node)
    # new variable
    if node[0] == 'var' and len(node) == 3:
        if not allow_var:
            raise ValueError('variable declaration not allowed here')
        return pl_comp_newvar(fenv, node)
    # write a variable
    if node[0] == 'set' and len(node) == 3:
        return pl_comp_setvar(fenv, node)
    # loop
    if node[0] == 'loop' and len(node) == 3:
        return pl_comp_loop(fenv, node)
    # break & continue
    if node[0] == 'break' and len(node) == 1:
        if fenv.scope.loop_end < 0:
            raise ValueError('`break` outside a loop')
        fenv.code.append(('jmp', fenv.scope.loop_end))
        return ('void'), -1
    if node[0] == 'continue' and len(node) == 1:
        if fenv.scope.loop_start < 0:
            raise ValueError('`continue` outside a loop')
        fenv.code.append(('jmp', fenv.scope.loop_start))
        return ('void'), -1
    # function definition
    if node[0] == 'def' and len(node) == 4:
        if not allow_var:
            raise ValueError('function not allowed here')
        return pl_comp_func(fenv, node)
    # function call
    if node[0] == 'call' and len(node) >= 2:
        return pl_comp_call(fenv, node)
    if node[0] == 'syscall' and len(node) >= 2:
        return pl_comp_syscall(fenv, node)
    # return
    if node[0] == 'return' and len(node) in (1, 2):
        return pl_comp_return(fenv, node)
    # null pointer
    if node[0] == 'ptr':
        tp = tuple(node)
        validate_type(tp)
        dst = fenv.tmp()
        fenv.code.append(('const', 0, dst))
        return tp, dst
    # cast
    if node[0] == 'cast' and len(node) == 3:
        return pl_comp_cast(fenv, node)
    # peek & poke
    if node[0] == 'peek' and len(node) == 2:
        return pl_comp_peek(fenv, node)
    if node[0] == 'poke' and len(node) == 3:
        return pl_comp_poke(fenv, node)
    # debug
    if node == ['debug']:
        fenv.code.append(('debug',))
        return ('void',), -1

    raise ValueError('unknown expression')


def pl_comp_cast(fenv: Func, node):
    _, tp, value = node
    tp = tuple(tp)
    validate_type(tp)
    val_tp, var = pl_comp_expr_tmp(fenv, value)

    # to, from
    free = [
        ('int', 'ptr'),
        ('ptr', 'int'),
        ('ptr', 'ptr'),
        ('int', 'byte'),
        ('int', 'int'),
        ('byte', 'byte'),
    ]
    if (tp[0], val_tp[0]) in free:
        return tp, var
    if (tp[0], val_tp[0]) == ('byte', 'int'):
        fenv.code.append(('cast8', var))
        return tp, var

    raise ValueError('bad cast')


def pl_comp_syscall(fenv: Func, node):
    _, num, *args = node
    if isinstance(num, list) and num[0] == 'val':
        _, num = num
    if not isinstance(num, int) or num < 0:
        raise ValueError('bad syscall number')

    save = fenv.stack
    sys_vars = []
    for kid in args:
        arg_tp, var = pl_comp_expr_tmp(fenv, kid)
        if arg_tp == ('void',):
            raise ValueError('bad syscall argument type')
        sys_vars.append(var)
    fenv.stack = save

    fenv.code.append(('syscall', fenv.stack, num, *sys_vars))
    return (('int',), fenv.tmp())


def pl_comp_peek(fenv: Func, node):
    _, ptr = node
    tp, var = pl_comp_expr(fenv, ptr)
    head, *tail = tp
    tail = tuple(tail)
    if head != 'ptr':
        raise ValueError('not a pointer')
    suffix = ''
    if tail == ('byte',):
        suffix = '8'
    fenv.code.append(('peek' + suffix, var, fenv.stack))
    return tail, fenv.tmp()


def pl_comp_poke(fenv: Func, node):
    _, ptr, value = node

    save = fenv.stack
    t2, var_val = pl_comp_expr_tmp(fenv, value)
    t1, var_ptr = pl_comp_expr_tmp(fenv, ptr)
    if t1 != ('ptr', *t2):
        raise ValueError('pointer type mismatch')
    fenv.stack = save

    suffix = ''
    if t2 == ('byte',):
        suffix = '8'
    fenv.code.append(('poke' + suffix, var_ptr, var_val))
    return t2, move_to(fenv, var_val, fenv.tmp())


def pl_comp_main(fenv: Func, node):
    assert node[:3] == ['def', ['main', 'int'], []]
    pl_scan_func(fenv, node)
    return pl_comp_func(fenv, node)


def pl_comp_return(fenv: Func, node):
    _, *kid = node
    tp, var = ('void',), -1
    if kid:
        tp, var = pl_comp_expr_tmp(fenv, kid[0])
    if tp != fenv.rtype:
        raise ValueError('bad return type')
    fenv.code.append(('ret', var))
    return tp, var


def pl_comp_call(fenv: Func, node):
    _, name, *args = node

    call_types = []
    for kid in args:
        arg_tp, var = pl_comp_expr(fenv, kid)
        call_types.append(arg_tp)
        move_to(fenv, var, fenv.tmp())

    key = (name, tuple(call_types))
    _, _, idx = fenv.get_var(key)
    func = fenv.root.funcs[idx]

    fenv.stack -= len(args)
    fenv.code.append(('call', idx, fenv.stack, fenv.level, func.level))
    dst = -1
    if func.rtype != ('void',):
        dst = fenv.tmp()
    return func.rtype, dst


def pl_comp_scope(fenv: Func, node):
    fenv.scope_enter()
    tp, var = ('void',), -1

    # separate kids by new variables
    groups = [[]]
    for kid in node[1:]:
        groups[-1].append(kid)
        if kid[0] == 'var':
            groups.append([])

    # Functions are visible before they are defined,
    # as long as they don't cross a variable.
    for g in groups:
        for kid in g:
            if kid[0] == 'def' and len(kid) == 4:
                pl_scan_func(fenv, kid)
        for kid in g:
            tp, var = pl_comp_expr(fenv, kid, allow_var=True)

    fenv.scope_leave()

    if var >= fenv.stack:
        var = move_to(fenv, var, fenv.tmp())
    return tp, var


def move_to(fenv, var, dst):
    if dst != var:
        fenv.code.append(('mov', var, dst))
    return dst


def pl_comp_newvar(fenv: Func, node):
    assert fenv.stack == fenv.nvar
    _, name, kid = node

    tp, var = pl_comp_expr(fenv, kid)
    if var < 0:
        raise ValueError('bad variable init type')

    fenv.add_var(name, tp)
    return tp, move_to(fenv, var, fenv.tmp())


def pl_comp_setvar(fenv: Func, node):
    _, name, kid = node

    tp, var = pl_comp_expr(fenv, kid)
    if var < 0:
        raise ValueError('bad variable set type')

    flevel, dst_tp, dst = fenv.get_var(name)
    if dst_tp != tp:
        raise ValueError('bad variable set type')

    if flevel == fenv.level:
        # local
        return dst_tp, move_to(fenv, var, dst)
    else:
        # outer
        fenv.code.append(('set_env', flevel, dst, var))
        return dst_tp, move_to(fenv, var, fenv.tmp())


def pl_comp_cond(fenv: Func, node):
    _, cond, yes, *no = node
    l_true = fenv.new_label()
    l_false = fenv.new_label()
    fenv.scope_enter()

    # cond
    tp, var = pl_comp_expr(fenv, cond, allow_var=True)
    if tp == ('void',):
        raise ValueError('expect boolean condition')
    fenv.code.append(('jmpf', var, l_false))

    # yes
    t1, a1 = pl_comp_expr(fenv, yes)
    if a1 >= 0:
        move_to(fenv, a1, fenv.stack)

    # no
    t2, a2 = ('void',), -1
    if no:
        fenv.code.append(('jmp', l_true))
    fenv.set_label(l_false)
    if no:
        t2, a2 = pl_comp_expr(fenv, no[0])
        if a2 >= 0:
            move_to(fenv, a2, fenv.stack)
    fenv.set_label(l_true)

    fenv.scope_leave()
    if a1 < 0 or a2 < 0 or t1 != t2:
        return ('void',), -1
    else:
        return t1, fenv.tmp()


def pl_comp_loop(fenv: Func, node):
    _, cond, body = node
    fenv.scope.loop_start = fenv.new_label()
    fenv.scope.loop_end = fenv.new_label()

    # enter
    fenv.scope_enter()
    fenv.set_label(fenv.scope.loop_start)
    # cond
    _, var = pl_comp_expr(fenv, cond, allow_var=True)
    if var < 0:
        raise ValueError('bad condition type')
    fenv.code.append(('jmpf', var, fenv.scope.loop_end))
    # body
    _, _ = pl_comp_expr(fenv, body)
    # loop
    fenv.code.append(('jmp', fenv.scope.loop_start))
    # leave
    fenv.set_label(fenv.scope.loop_end)
    fenv.scope_leave()

    return ('void',), -1


def validate_type(tp):
    assert isinstance(tp, tuple)
    if len(tp) == 0:
        raise ValueError('type missing')
    head, *body = tp
    body = tuple(body)
    if head == 'ptr':
        if not body or body == ('void',):
            raise ValueError('bad pointer element')
        validate_type(body)
    elif head in ('void', 'int', 'byte'):
        if body:
            raise ValueError('bad scaler type')
    else:
        raise ValueError('unknown type')


# make the function visible to the whole scope before its definition.
def pl_scan_func(fenv: Func, node):
    _, (name, *rtype), args, _ = node
    rtype = tuple(rtype)
    validate_type(rtype)

    arg_type_list = tuple(tuple(arg_type) for _, *arg_type in args)
    key = (name, arg_type_list)
    if key in fenv.scope.names:
        raise ValueError('duplicated function')

    fenv.scope.names[key] = (rtype, len(fenv.root.funcs))
    fenv = Func(fenv)
    fenv.rtype = rtype
    fenv.root.funcs.append(fenv)


def pl_comp_func(fenv: Func, node):
    _, (name, *_), args, body = node
    arg_type_list = tuple(tuple(arg_type) for _, *arg_type in args)
    key = (name, arg_type_list)
    rtype, idx = fenv.scope.names[key]
    fenv = fenv.root.funcs[idx]

    for arg_name, *arg_type in args:
        if not isinstance(arg_name, str):
            raise ValueError('bad argument name')
        arg_type = tuple(arg_type)
        validate_type(arg_type)
        if arg_type == ('void',):
            raise ValueError('bad argument type')
        fenv.add_var(arg_name, arg_type)
    fenv.stack = len(args)

    body_type, var = pl_comp_expr(fenv, body)
    if rtype != ('void',) and rtype != body_type:
        raise ValueError('bad body type')
    if rtype == ('void',):
        var = -1
    fenv.code.append(('ret', var))

    fenv = fenv.prev
    return ('void',), -1


# dissambler:
# objdump -b binary -M intel,x86-64 -m i386 --adjust-vma=0x1000 --start-address=0x1080
class CodeGen:
    A = 0
    C = 1
    D = 2
    B = 3
    SP = 4
    BP = 5
    SI = 6
    DI = 7

    def __init__(self):
        # params
        self.vaddr = 0x1000
        self.alignment = 16
        # output
        self.buf = bytearray()
        # states
        self.jmps = dict()
        self.calls = dict()
        self.strings = dict()
        self.func2off = []
        self.fields = dict()

    def f16(self, name):
        self.fields[name] = (16, len(self.buf))
        self.buf.extend(b'\0\0')
    def f32(self, name):
        self.fields[name] = (32, len(self.buf))
        self.buf.extend(b'\0\0\0\0')
    def f64(self, name):
        self.fields[name] = (64, len(self.buf))
        self.buf.extend(b'\0' * 8)

    def setf(self, name, i):
        bits, off = self.fields[name]
        fmt = {16: '<H', 32: '<I', 64: '<Q'}[bits]
        sz = bits // 8
        self.buf[off:off+sz] = struct.pack(fmt, i)

    def elf_begin(self):
        self.elf_header()

        phdr_start = len(self.buf)
        self.elf_program_header()
        self.setf('e_phentsize', len(self.buf) - phdr_start)
        self.setf('e_phnum', 1)

        self.padding()
        self.setf('e_entry', self.vaddr + len(self.buf))

    def elf_header(self):
        # ref: https://www.muppetlabs.com/~breadbox/software/tiny/tiny-elf64.asm.txt
        self.buf.extend(bytes.fromhex('7F 45 4C 46 02 01 01 00 00 00 00 00 00 00 00 00'))
        # e_type, e_machine, e_version
        self.buf.extend(bytes.fromhex('02 00 3E 00 01 00 00 00'))
        self.f64('e_entry')
        self.f64('e_phoff')
        self.f64('e_shoff')
        self.f32('e_flags')
        self.f16('e_ehsize')
        self.f16('e_phentsize')
        self.f16('e_phnum')
        self.f16('e_shentsize')
        self.f16('e_shnum')
        self.f16('e_shstrndx')
        self.setf('e_phoff', len(self.buf))
        self.setf('e_ehsize', len(self.buf))

    def elf_program_header(self):
        # p_type, p_flags, p_offset
        self.buf.extend(bytes.fromhex('01 00 00 00 05 00 00 00 00 00 00 00 00 00 00 00'))
        # p_vaddr, p_paddr
        self.i64(self.vaddr)
        self.i64(self.vaddr)
        self.f64('p_filesz')
        self.f64('p_memsz')
        # p_align
        self.i64(0x1000)

    def output_elf(self, root: Func):
        self.elf_begin()
        self.code_entry()
        for func in root.funcs:
            self.func(func)
        self.code_end()
        self.elf_end()

    def elf_end(self):
        # program header
        self.setf('p_filesz', len(self.buf))
        self.setf('p_memsz', len(self.buf))

    def create_stack(self, data):
        def operand(i):
            return struct.pack('<i', i)

        # syscall ref: https://blog.rchapman.org/posts/Linux_System_Call_Table_for_x86_64/
        # syscall abi: https://github.com/torvalds/linux/blob/v5.0/arch/x86/entry/entry_64.S#L107
        # mmap
        self.buf.extend(
            b"\xb8\x09\x00\x00\x00"         # mov eax, 9
            # b"\x31\xff"                     # xor edi, edi      // addr = NULL
            b"\xbf\x00\x10\x00\x00"         # mov edi, 4096     // addr
            b"\x48\xc7\xc6%s"               # mov rsi, xxx      // len
            b"\xba\x03\x00\x00\x00"         # mov edx, 3        // prot = PROT_READ|PROT_WRITE
            b"\x41\xba\x22\x00\x00\x00"     # mov r10d, 0x22    // flags = MAP_PRIVATE|MAP_ANONYMOUS
            b"\x49\x83\xc8\xff"             # or r8, -1         // fd = -1
            b"\x4d\x31\xc9"                 # xor r9, r9        // offset = 0
            b"\x0f\x05"                     # syscall
            % (operand(data + 4096),)
        )

        # mprotect
        self.buf.extend(
            b"\x48\x89\xc3"                 # mov rbx, rax      // the data stack
            b"\xb8\x0a\x00\x00\x00"         # mov eax, 10
            b"\x48\x8d\xbb%s"               # lea rdi, [rbx + data]
            b"\xbe\x00\x10\x00\x00"         # mov esi, 4096
            b"\x31\xd2"                     # xor edx, edx
            b"\x0f\x05"                     # syscall
            % (operand(data))
        )
        # FIXME: check the syscall return value

    def code_entry(self):
        # create the data stack (8M)
        self.create_stack(0x800000)
        # call the main function
        self.asm_call(0)
        # exit
        self.buf.extend(
            b"\xb8\x3c\x00\x00\x00"         # mov eax, 60
            b"\x48\x8b\x3b"                 # mov rdi, [rbx]
            b"\x0f\x05"                     # syscall
        )

    def padding(self):
        if self.alignment == 0:
            return
        while len(self.buf) % self.alignment:
            self.buf.append(0xcc)           # int3

    def func(self, func: Func):
        # alignment
        self.padding()

        # offsets
        self.func2off.append(len(self.buf))
        pos2off = []

        # each instruction
        for instr_name, *instr_args in func.code:
            pos2off.append(len(self.buf))
            method = getattr(self.__class__, instr_name)
            method(self, *instr_args)

        # fill the jmp address
        for L, off_list in self.jmps.items():
            dst_off = pos2off[func.labels[L]]
            for patch_off in off_list:
                self.patch_addr(patch_off, dst_off)
        self.jmps.clear()

    def patch_addr(self, patch_off, dst_off):
        src_off = patch_off + 4
        relative = struct.pack('<i', dst_off - src_off)
        self.buf[patch_off:patch_off+4] = relative

    def code_end(self):
        # fill the call address
        for L, off_list in self.calls.items():
            dst_off = self.func2off[L]
            for patch_off in off_list:
                self.patch_addr(patch_off, dst_off)
        self.calls.clear()
        # alignment
        self.padding()
        # strings
        for s, off_list in self.strings.items():
            dst_off = len(self.buf)
            for patch_off in off_list:
                self.patch_addr(patch_off, dst_off)
            self.buf.extend(s.encode('utf-8') + b'\0')
        self.strings.clear()
        # alignment
        self.padding()

    def i32(self, i):
        self.buf.extend(struct.pack('<i', i))

    def i64(self, i):
        self.buf.extend(struct.pack('<q', i))

    def asm_disp(self, opcode, reg, ptr, off):
        assert reg < 16 and ptr < 16
        if reg >= 8 or ptr >= 8:
            assert (opcode[0] >> 4) == 0b0100

        opcode = bytearray(opcode)
        opcode[0] |= (reg >> 3) << 2    # REX.R
        opcode[0] |= (ptr >> 3) << 0    # REX.B
        reg &= 0b111
        ptr &= 0b111

        self.buf.extend(opcode)
        if off == 0:
            mod = 0
        elif -128 <= off < 128:
            mod = 1
        else:
            mod = 3
        self.buf.append((mod << 6) | (reg << 3) | ptr)
        if mod == 1:
            self.buf.append(off if off > 0 else (256 + off))
        if mod == 3:
            self.i32(off)

    def asm_load(self, reg, ptr, off):
        self.asm_disp(b'\x48\x8b', reg, ptr, off)

    def asm_store(self, ptr, off, reg):
        self.asm_disp(b'\x48\x89', reg, ptr, off)

    def store_rax(self, dst):
        # mov [rbx + dst32*8], rax
        self.asm_store(CodeGen.B, dst * 8, CodeGen.A)

    def load_rax(self, src):
        # mov rax, [rbx + src32*8]
        self.asm_load(CodeGen.A, CodeGen.B, src * 8)

    def const(self, val, dst):
        assert isinstance(val, (int, str))
        if isinstance(val, str):
            # lea rax, [rip + offset]
            self.buf.extend(b"\x48\x8d\x05")
            self.strings.setdefault(val, []).append(len(self.buf))
            self.buf.extend(b"\0\0\0\0")
        elif val == 0:
            self.buf.extend(b"\x31\xc0")            # xor eax, eax
        elif val == -1:
            self.buf.extend(b"\x48\x83\xc8\xff")    # or rax, -1
        elif (val >> 31) == 0:
            self.buf.extend(b"\xb8")                # mov, eax, imm32
            self.i32(val)
        elif (val >> 31) == -1:
            # sign-extended
            self.buf.extend(b"\x48\xc7\xc0")        # mov, rax, imm32
            self.i32(val)
        else:
            self.buf.extend(b'\x48\xb8')            # mov rax, imm64
            self.i64(val)
        self.store_rax(dst)

    def mov(self, src, dst):
        if src == dst:
            return
        self.load_rax(src)
        self.store_rax(dst)

    def binop(self, op, a1, a2, dst):
        self.load_rax(a1)

        arith = {
            '+': b'\x48\x03\x83',
            '-': b'\x48\x2b\x83',
            '*': b'\x48\x0f\xaf\x83',
        }
        cmp = {
            'eq': b'\x0f\x94\xc0',
            'ne': b'\x0f\x95\xc0',
            'ge': b'\x0f\x9d\xc0',
            'gt': b'\x0f\x9f\xc0',
            'le': b'\x0f\x9e\xc0',
            'lt': b'\x0f\x9c\xc0',
        }

        if op in ('/', '%'):
            # xor edx, edx
            self.buf.extend(b"\x31\xd2")
            # idiv rax, [rbx + a2*8]
            self.buf.extend(b'\x48\xf7\xbb')
            self.i32(a2 * 8)
            if op == '%':
                # mov, rax, rdx
                self.buf.extend(b"\x48\x89\xd0")
        elif op in arith:
            self.asm_disp(arith[op][:-1], CodeGen.A, CodeGen.B, a2 * 8)
        elif op in cmp:
            # cmp rax, [rbx + a2*8]
            self.asm_disp(b'\x48\x3b', CodeGen.A, CodeGen.B, a2 * 8)
            # setcc al
            self.buf.extend(cmp[op])
            # movzx eax, al
            self.buf.extend(b"\x0f\xb6\xc0")
        elif op == 'and':
            self.buf.extend(
                b"\x48\x85\xc0"     # test rax, rax
                b"\x0f\x95\xc0"     # setne al
            )
            # mov rdx, [rbx + a2*8]
            self.asm_load(CodeGen.D, CodeGen.B, a2 * 8)
            self.buf.extend(
                b"\x48\x85\xd2"     # test rdx, rdx
                b"\x0f\x95\xc2"     # setne dl
                b"\x21\xd0"         # and eax, edx
                b"\x0f\xb6\xc0"     # movzx eax, al
            )
        elif op == 'or':
            # or rax, [rbx + a2*8]
            self.asm_disp(b"\x48\x0b", CodeGen.A, CodeGen.B, a2 * 8)
            self.buf.extend(
                b"\x0f\x95\xc0"     # setne al
                b"\x0f\xb6\xc0"     # movzx eax, al
            )
        else:
            raise NotImplementedError

        self.store_rax(dst)

    def unop(self, op, a1, dst):
        self.load_rax(a1)
        if op == '-':
            self.buf.extend(b"\x48\xf7\xd8")    # neg rax
        elif op == 'not':
            self.buf.extend(
                b"\x48\x85\xc0"     # test rax, rax
                b"\x0f\x94\xc0"     # sete al
                b"\x0f\xb6\xc0"     # movzx eax, al
            )
        else:
            raise NotImplementedError
        self.store_rax(dst)

    def jmpf(self, a1, L):
        self.load_rax(a1)
        self.buf.extend(
            b"\x48\x85\xc0"         # test rax, rax
            b"\x0f\x84"             # je
        )
        self.jmps.setdefault(L, []).append(len(self.buf))
        self.buf.extend(b'\0\0\0\0')

    def jmp(self, L):
        self.buf.extend(b"\xe9")    # jmp
        self.jmps.setdefault(L, []).append(len(self.buf))
        self.buf.extend(b'\0\0\0\0')

    def asm_call(self, L):
        self.buf.extend(b"\xe8")    # call
        self.calls.setdefault(L, []).append(len(self.buf))
        self.buf.extend(b'\0\0\0\0')

    def ret(self, a1):
        if a1 > 0:
            self.load_rax(a1)
            self.store_rax(0)
        self.buf.append(0xc3)       # ret

    def asm_op32_nonzero(self, prefix, i):
        if i != 0:
            self.buf.extend(prefix)
            self.i32(i)

    def call(self, func, arg_start, level_cur, level_new):
        assert 1 <= level_cur
        assert 1 <= level_new <= level_cur + 1

        # pointers to outer frames
        if level_new > level_cur:
            self.buf.append(0x53)               # push rbx
        for _ in range(min(level_new, level_cur) - 1):
            self.buf.extend(b"\xff\xb4\x24")    # push [rsp + (level_new - 1)*8]
            self.i32((level_new - 1) * 8)

        # make a new frame and call the target
        self.asm_op32_nonzero(b"\x48\x81\xc3",  # add rbx, arg_start*8
            arg_start * 8)
        self.asm_call(func)                     # call func
        self.asm_op32_nonzero(b"\x48\x81\xc3",  # add rbx, -arg_start*8
            -arg_start * 8)

        # cleanups
        self.buf.extend(b"\x48\x81\xc4")        # add rsp, (level_new - 1)*8
        self.i32((level_new - 1) * 8)

    def load_env_addr(self, level_var):
        self.buf.extend(b"\x48\x8b\x84\x24")    # mov rax, [rsp + level_var*8]
        self.i32(level_var * 8)

    def get_env(self, level_var, var, dst):
        self.load_env_addr(level_var)
        # mov rax, [rax + var*8]
        self.asm_load(CodeGen.A, CodeGen.A, var * 8)
        # mov [rbx + dst*8], rax
        self.store_rax(dst)

    def set_env(self, level_var, var, src):
        self.load_env_addr(level_var)
        # mov rdx, [rbx + src*8]
        self.asm_load(CodeGen.D, CodeGen.B, src * 8)
        # mov [rax + var*8], rdx
        self.asm_store(CodeGen.A, var * 8, CodeGen.D)

    def lea(self, a1, a2, scale, dst):
        self.load_rax(a1)
        self.asm_load(CodeGen.D, CodeGen.B, a2 * 8) # mov rdx, [rbx + a2*8]
        if scale < 0:
            self.buf.extend(b"\x48\xf7\xda")        # neg rdx
        self.buf.extend({
            1: b"\x48\x8d\x04\x10",                 # lea rax, [rax + rdx]
            2: b"\x48\x8d\x04\x50",                 # lea rax, [rax + rdx*2]
            4: b"\x48\x8d\x04\x90",                 # lea rax, [rax + rdx*4]
            8: b"\x48\x8d\x04\xd0",                 # lea rax, [rax + rdx*8]
        }[abs(scale)])
        self.store_rax(dst)

    def peek(self, var, dst):
        self.load_rax(var)
        # mov rax, [rax]
        self.asm_load(CodeGen.A, CodeGen.A, 0)
        self.store_rax(dst)

    def peek8(self, var, dst):
        self.load_rax(var)
        # movzx eax, byte ptr [rax]
        self.buf.extend(b"\x0f\xb6\x00")
        self.store_rax(dst)

    def poke(self, ptr, val):
        self.load_rax(val)
        # mov rdx, [rbx + ptr*8]
        self.asm_load(CodeGen.D, CodeGen.B, ptr * 8)
        # mov [rdx], rax
        self.asm_store(CodeGen.D, 0, CodeGen.A)

    def poke8(self, ptr, val):
        self.load_rax(val)
        # mov rdx, [rbx + ptr*8]
        self.asm_load(CodeGen.D, CodeGen.B, ptr * 8)
        # mov [rdx], al
        self.buf.extend(b"\x88\x02")

    def cast8(self, var):
        # movzx eax, byte ptr [rbx + var*8]
        self.asm_disp(b"\x0f\xb6", CodeGen.A, CodeGen.B, var * 8)
        self.store_rax(var)

    def syscall(self, dst, num, *arg_list):
        # syscall ref: https://blog.rchapman.org/posts/Linux_System_Call_Table_for_x86_64/
        self.buf.extend(b"\xb8")                # mov, eax, imm32
        self.i32(num)
        arg_regs = [CodeGen.DI, CodeGen.SI, CodeGen.D, 10, 8, 9]
        assert len(arg_list) <= len(arg_regs)
        for i, arg in enumerate(arg_list):
            # mov reg, [rbx + arg*8]
            self.asm_load(arg_regs[i], CodeGen.B, arg * 8)
        self.buf.extend(b"\x0f\x05")            # syscall
        self.store_rax(dst)                     # mov [rbx + dst*8], rax

    def debug(self):
        self.buf.append(0xcc)                   # int3


# ir
'''
const val dst
mov src dst
binop op a1 a2 dst
unop op a1 dst
binop8 op a1 a2 dst
unop8 op a1 dst
jmpf a1 L
jmp L
ret a1
ret -1
call func arg_start level_cur level_new
get_env level_var var dst
set_env level_var var src
lea
peek
poke
peek8
poke8
cast8
syscall
debug
'''


# syntax
'''
(+ a b)
(- a b)
(* a b)
(/ a b)
...

(eq a b)
(ne a b)
(ge a b)
(gt a b)
(le a b)
(lt a b)

(not b)
(and a b)
(or a b)

(? cond yes no)
(if cond (then yes blah blah) (else no no no))
(do a b c...)
(var name val)
(set name val)
(loop cond body)
(break)
(continue)

(def (name rtype) ((a1 a1type) (a2 a2type)...) body)
(call f a b c...)
(return val)

(ptr elem_type)
(peek ptr)
(poke ptr value)
(syscall num args...)
(cast type val)
'''


# types
'''
void
int
byte
ptr int
ptr byte
'''


def ir_dump(root: Func):
    out = []
    for i, func in enumerate(root.funcs):
        out.append(f'func{i}:')
        pos2labels = dict()
        for label, pos in func.labels.items():
            pos2labels.setdefault(pos, []).append(label)
        for pos, instr in enumerate(func.code):
            for label in pos2labels.get(pos, []):
                out.append(f'L{label}:')
            if instr[0].startswith('jmp'):
                instr = instr[:-1] + (f'L{instr[-1]}',)
            if instr[0] == 'const' and isinstance(instr[1], str):
                import json
                instr = list(instr)
                instr[1] = json.dumps(instr[1])
            out.append('    ' + ' '.join(map(str, instr)))
        out.append('')

    return '\n'.join(out)


def test_comp():
    def f(s):
        node = pl_parse_main(s)
        fenv = Func(None)
        pl_comp_main(fenv, node)
        return [x.code for x in fenv.funcs]

    def asm(s):
        node = pl_parse_main(s)
        fenv = Func(None)
        pl_comp_main(fenv, node)
        return ir_dump(fenv)

    assert f('1') == [[
        ('const', 1, 0),
        ('ret', 0),
    ]]
    assert f('1 3') == [[
        ('const', 1, 0),
        ('const', 3, 0),
        ('ret', 0),
    ]]
    assert f('(+ (- 1 2) 3)') == [[
        ('const', 1, 0),
        ('const', 2, 1),
        ('binop', '-', 0, 1, 0),
        ('const', 3, 1),
        ('binop', '+', 0, 1, 0),
        ('ret', 0),
    ]]
    assert f('(return 1)') == [[
        ('const', 1, 0),
        ('ret', 0),
        ('ret', 0),
    ]]
    assert asm('(if 1 2 3)').split() == '''
        func0:
            const 1 0
            jmpf 0 L1
            const 2 0
            jmp L0
        L1:
            const 3 0
        L0:
            ret 0
    '''.split()
    assert asm('''
        (loop (var a 1) (do
            (var b a)
            (if (gt a 11)
                (break))
            (var c (set a (+ 2 b)))
            (if (lt c 100)
                (continue))
            (set b 5)
        ))
        0''').split() == '''
        func0:
        L0:
            const 1 0
            jmpf 0 L1
            mov 0 1
            const 11 2
            binop gt 0 2 2
            jmpf 2 L3
            jmp L1
        L2:
        L3:
            const 2 2
            binop + 2 1 2
            mov 2 0
            mov 0 2
            const 100 3
            binop lt 2 3 3
            jmpf 3 L5
            jmp L0
        L4:
        L5:
            const 5 3
            mov 3 1
            jmp L0
        L1:
            const 0 0
            ret 0
    '''.split()
    assert asm('(if 1 (return 2)) 0').split() == '''
        func0:
            const 1 0
            jmpf 0 L1
            const 2 0
            ret 0
        L0:
        L1:
            const 0 0
            ret 0
    '''.split()
    assert asm('(var a 1) (set a (+ 3 a)) (var b 2) (- b a)').split() == '''
        func0:
            const 1 0
            const 3 1
            binop + 1 0 1
            mov 1 0
            const 2 1
            binop - 1 0 2
            mov 2 0
            ret 0
    '''.split()
    assert asm('(var a 1) (return (+ 3 a))').split() == '''
        func0:
            const 1 0
            const 3 1
            binop + 1 0 1
            ret 1
            mov 1 0
            ret 0
    '''.split()
    assert asm('(var a 1) (+ 3 a)').split() == '''
        func0:
            const 1 0
            const 3 1
            binop + 1 0 1
            mov 1 0
            ret 0
    '''.split()
    assert asm('''
        (def (fib int) ((n int)) (if (le n 0) (then 0) (else (call fib (- n 1)))))
        (call fib 5)
        ''').split() == '''
        func0:
            const 5 0
            call 1 0 1 2
            ret 0
        func1:
            const 0 1
            binop le 0 1 1
            jmpf 1 L1
            const 0 1
            jmp L0
        L1:
            const 1 1
            binop - 0 1 1
            call 1 1 2 2
        L0:
            ret 1
    '''.split()
    assert asm('''
        (var b 456)
        (def (f void) () (do
            (var a 123)
            (def (g void) () (do
                (set a (+ b a))
            ))
            (call g)
        ))

        (call f)
        0
        ''').split() == '''
        func0:
            const 456 0
            call 1 1 1 2
            const 0 1
            mov 1 0
            ret 0
        func1:
            const 123 0
            call 2 1 2 3
            ret -1
        func2:
            get_env 1 0 0
            get_env 2 0 1
            binop + 0 1 0
            set_env 2 0 0
            ret -1
    '''.split()
    assert asm('''
        (var p (ptr int))
        (poke (cast (ptr byte) p) 124u8)
        (peek (cast (ptr byte) p))
        (poke p 123)
    ''').split() == '''
        func0:
            const 0 0
            const 124 1
            poke8 0 1
            peek8 0 1
            const 123 1
            poke 0 1
            mov 1 0
            ret 0
    '''.split()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('file', nargs='?', help='the input source file')
    ap.add_argument('-o', '--output', help='the output path')
    ap.add_argument('--print-ir', action='store_true', help='print the intermediate representation')
    ap.add_argument('--alignment', type=int, default=16)
    args = ap.parse_args()
    if not args.file or not args.output:
        ap.print_help()
        test_comp()
        return

    with open(args.file, 'rt', encoding='utf-8') as fp:
        text = fp.read()

    node = pl_parse_main(text)
    main = Func(None)
    _ = pl_comp_main(main, node)
    if args.print_ir:
        print(ir_dump(main))

    gen = CodeGen()
    gen.alignment = args.alignment
    gen.output_elf(main)

    import os
    fd = os.open(args.output, os.O_WRONLY|os.O_CREAT|os.O_TRUNC, 0o755)
    with os.fdopen(fd, 'wb', closefd=True) as fp:
        fp.write(gen.buf)


if __name__ == '__main__':
    main()
