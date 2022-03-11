from collections import namedtuple, OrderedDict
import dis
import inspect
import itertools
from types import CodeType, ModuleType

from numba.core import errors, utils, serialize


opcode_info = namedtuple('opcode_info', ['argsize'])

# The following offset is used as a hack to inject a NOP at the start of the
# bytecode. So that function starting with `while True` will not have block-0
# as a jump target. The Lowerer puts argument initialization at block-0.
_FIXED_OFFSET = 0


def get_function_object(obj):
    """
    Objects that wraps function should provide a "__numba__" magic attribute
    that contains a name of an attribute that contains the actual python
    function object.
    """
    attr = getattr(obj, "__numba__", None)
    if attr:
        return getattr(obj, attr)
    return obj


def get_code_object(obj):
    "Shamelessly borrowed from llpython"
    return getattr(obj, '__code__', getattr(obj, 'func_code', None))


def _as_opcodes(seq):
    lst = []
    for s in seq:
        c = dis.opmap.get(s)
        if c is not None:
            lst.append(c.opcode)
    return lst


# JREL_OPS = frozenset(dis.hasjrel)
# JABS_OPS = frozenset(dis.hasjabs)
JUMP_OPS = frozenset(op.opcode for op in dis.bytecodes if op.is_jump())
# JUMP_OPS = JREL_OPS | JABS_OPS
TERM_OPS = frozenset(_as_opcodes(['RETURN_VALUE', 'RAISE']))
JUMP_IMM1 = frozenset(_as_opcodes(['FOR_ITER', 'JUMP_IF_NOT_EXC_MATCH']))
# EXTENDED_ARG = dis.EXTENDED_ARG
# HAVE_ARGUMENT = dis.HAVE_ARGUMENT


class ByteCodeInst(object):
    '''
    Attributes
    ----------
    - offset:
        byte offset of opcode
    - opcode:
        opcode integer value
    - arg:
        instruction arg
    - lineno:
        -1 means unknown
    '''
    __slots__ = 'offset', 'next', 'opcode', 'opname', 'arg', 'lineno', 'imm'

    def __init__(self, instr):
        self.offset = instr.offset
        self.opcode = instr.opcode
        self.opname = dis.opname[self.opcode]
        self.imm = instr.imm
        self.arg = instr.imm[0] if len(instr.imm) > 0 else None
        self.lineno = -1  # unknown line number

    @property
    def is_jump(self):
        return self.opcode in JUMP_OPS

    @property
    def is_terminator(self):
        return self.opcode in TERM_OPS

    def get_jump_target(self):
        assert self.is_jump
        if self.opcode in JUMP_IMM1:
            return self.offset + self.imm[1]
        else:
            return self.offset + self.imm[0]

    def __repr__(self):
        if len(self.imm) == 0:
            return '%s(lineno=%d)' % (self.opname, self.lineno)
        elif len(self.imm) == 1:
            return '%s(arg=%s, lineno=%d)' % (self.opname, self.arg, self.lineno)
        else:
            return '%s(args=%s, lineno=%d)' % (self.opname, str(self.imm), self.lineno)


class ByteCodeIter(object):
    def __init__(self, code):
        self.code = code
        self.instrs = [ByteCodeInst(i) for i in dis.get_instructions(self.code)]
        for bi1, bi2 in zip(self.instrs[:-1], self.instrs[1:]):
            bi1.next = bi2.offset
        self.iter = iter(self.instrs)

    def __iter__(self):
        return self

    def next(self):
        b = next(self.iter)
        return b.offset, b

    __next__ = next


class ByteCode(object):
    """
    The decoded bytecode of a function, and related information.
    """
    __slots__ = ('func_id', 'co_names', 'co_varnames', 'co_consts',
                 'co_cellvars', 'co_freevars', 'co_free2reg', 'num_locals', 'table', 'labels')

    def __init__(self, func_id):
        code = func_id.code

        labels = set(x + _FIXED_OFFSET for x in dis.findlabels(code.co_code))
        labels.add(0)

        # A map of {offset: ByteCodeInst}
        table = OrderedDict(ByteCodeIter(code))
        self._compute_lineno(table, code)

        self.func_id = func_id
        self.co_names = code.co_names
        self.co_varnames = code.co_varnames
        self.co_consts = code.co_consts
        self.co_cellvars = code.co_cellvars
        self.co_freevars = code.co_freevars
        self.co_free2reg = code.co_free2reg
        self.num_locals = len(self.co_varnames)
        self.table = table
        self.labels = sorted(labels)

        ntemp = code.co_framesize - len(self.co_varnames)
        self.co_varnames += tuple(f'.t{i}' for i in range(ntemp))

    @classmethod
    def _compute_lineno(cls, table, code):
        """
        Compute the line numbers for all bytecode instructions.
        """
        for offset, lineno in dis.findlinestarts(code):
            adj_offset = offset + _FIXED_OFFSET
            if adj_offset in table:
                table[adj_offset].lineno = lineno
        # Assign unfilled lineno
        # Start with first bytecode's lineno
        known = table[_FIXED_OFFSET].lineno
        for inst in table.values():
            if inst.lineno >= 0:
                known = inst.lineno
            else:
                inst.lineno = known
        return table

    def __iter__(self):
        return iter(self.table.values())

    def __getitem__(self, offset):
        return self.table[offset]

    def __contains__(self, offset):
        return offset in self.table

    def dump(self):
        def label_marker(i):
            if i[1].offset in self.labels:
                return '>'
            else:
                return ' '

        return '\n'.join('%s %10s\t%s' % ((label_marker(i),) + i)
                         for i in self.table.items())

    @classmethod
    def _compute_used_globals(cls, func, table, co_consts, co_names):
        """
        Compute the globals used by the function with the given
        bytecode table.
        """
        d = {}
        globs = func.__globals__
        builtins = globs.get('__builtins__', utils.builtins)
        if isinstance(builtins, ModuleType):
            builtins = builtins.__dict__
        # Look for LOAD_GLOBALs in the bytecode
        for inst in table.values():
            if inst.opname == 'LOAD_GLOBAL':
                name = co_names[inst.arg]
                if name not in d:
                    try:
                        value = globs[name]
                    except KeyError:
                        value = builtins[name]
                    d[name] = value
        # Add globals used by any nested code object
        for co in co_consts:
            if isinstance(co, CodeType):
                subtable = OrderedDict(ByteCodeIter(co))
                d.update(cls._compute_used_globals(func, subtable,
                                                   co.co_consts, co.co_names))
        return d

    def get_used_globals(self):
        """
        Get a {name: value} map of the globals used by this code
        object and any nested code objects.
        """
        return self._compute_used_globals(self.func_id.func, self.table,
                                          self.co_consts, self.co_names)


class FunctionIdentity(serialize.ReduceMixin):
    """
    A function's identity and metadata.

    Note this typically represents a function whose bytecode is
    being compiled, not necessarily the top-level user function
    (the two might be distinct, e.g. in the `@generated_jit` case).
    """
    _unique_ids = itertools.count(1)

    @classmethod
    def from_function(cls, pyfunc):
        """
        Create the FunctionIdentity of the given function.
        """
        func = get_function_object(pyfunc)
        code = get_code_object(func)
        pysig = utils.pysignature(func)
        if not code:
            raise errors.ByteCodeSupportError(
                "%s does not provide its bytecode" % func)

        try:
            func_qualname = func.__qualname__
        except AttributeError:
            func_qualname = func.__name__

        self = cls()
        self.func = func
        self.func_qualname = func_qualname
        self.func_name = func_qualname.split('.')[-1]
        self.code = code
        self.module = inspect.getmodule(func)
        self.modname = (utils._dynamic_modname
                        if self.module is None
                        else self.module.__name__)
        self.is_generator = inspect.isgeneratorfunction(func)
        self.pysig = pysig
        self.filename = code.co_filename
        self.firstlineno = code.co_firstlineno
        self.arg_count = len(pysig.parameters)
        self.arg_names = list(pysig.parameters)

        # Even the same function definition can be compiled into
        # several different function objects with distinct closure
        # variables, so we make sure to disambiguate using an unique id.
        uid = next(cls._unique_ids)
        self.unique_name = '{}${}'.format(self.func_qualname, uid)

        return self

    def derive(self):
        """Copy the object and increment the unique counter.
        """
        return self.from_function(self.func)

    def _reduce_states(self):
        """
        NOTE: part of ReduceMixin protocol
        """
        return dict(pyfunc=self.func)

    @classmethod
    def _rebuild(cls, pyfunc):
        """
        NOTE: part of ReduceMixin protocol
        """
        return cls.from_function(pyfunc)
