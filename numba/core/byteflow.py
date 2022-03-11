"""
Implement python 3.8+ bytecode analysis
"""

from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
import opcode

from numba.core.utils import UniqueDict, PYVERSION
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError


_logger = logging.getLogger(__name__)


_EXCEPT_STACK_OFFSET = 6
_FINALLY_POP = _EXCEPT_STACK_OFFSET if PYVERSION >= (3, 8) else 1
_NO_RAISE_OPS = frozenset({
    'LOAD_CONST',
})


@total_ordering
class BlockKind(object):
    """Kinds of block to make related code safer than just `str`.
    """
    _members = frozenset({
        'LOOP',
        'TRY', 'EXCEPT', 'FINALLY',
        'WITH', 'WITH_FINALLY',
    })

    def __init__(self, value):
        assert value in self._members
        self._value = value

    def __hash__(self):
        return hash((type(self), self._value))

    def __lt__(self, other):
        if isinstance(other, BlockKind):
            return self._value < other._value
        else:
            raise TypeError('cannot compare to {!r}'.format(type(other)))

    def __eq__(self, other):
        if isinstance(other, BlockKind):
            return self._value == other._value
        else:
            raise TypeError('cannot compare to {!r}'.format(type(other)))

    def __repr__(self):
        return "BlockKind({})".format(self._value)


class _lazy_pformat(object):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        return pformat(*self.args, **self.kwargs)


class Flow(object):
    """Data+Control Flow analysis.

    Simulate execution to recover dataflow and controlflow information.
    """
    def __init__(self, bytecode):
        _logger.debug("bytecode dump:\n%s", bytecode.dump())
        self._bytecode = bytecode
        self.block_infos = UniqueDict()

    def run(self):
        """Run a trace over the bytecode over all reachable path.

        The trace starts at bytecode offset 0 and gathers stack and control-
        flow information by partially interpreting each bytecode.
        Each ``State`` instance in the trace corresponds to a basic-block.
        The State instances forks when a jump instruction is encountered.
        A newly forked state is then added to the list of pending states.
        The trace ends when there are no more pending states.
        """
        firststate = State(bytecode=self._bytecode, pc=0)
        runner = TraceRunner(debug_filename=self._bytecode.func_id.filename)
        runner.pending.append(firststate)

        # Enforce unique-ness on initial PC to avoid re-entering the PC with
        # a different stack-depth. We don't know if such a case is ever
        # possible, but no such case has been encountered in our tests.
        first_encounter = UniqueDict()
        # Loop over each pending state at a initial PC.
        # Each state is tracing a basic block
        while runner.pending:
            _logger.debug("pending: %s", runner.pending)
            state = runner.pending.popleft()
            if state not in runner.finished:
                _logger.debug("stack: %s", state._stack)
                first_encounter[state.pc_initial] = state
                # Loop over the state until it is terminated.
                while True:
                    runner.dispatch(state)
                    # Terminated?
                    if state.has_terminated():
                        break
                    elif (state.has_active_try() and
                            state.get_inst().opname not in _NO_RAISE_OPS):
                        # Is in a *try* block
                        state.fork(pc=state.get_inst().next)
                        tryblk = state.get_top_block('TRY')
                        state.pop_block_and_above(tryblk)
                        nstack = state.stack_depth
                        kwargs = {}
                        if nstack > tryblk['entry_stack']:
                            kwargs['npop'] = nstack - tryblk['entry_stack']
                        handler = tryblk['handler']
                        kwargs['npush'] = {
                            BlockKind('EXCEPT'): _EXCEPT_STACK_OFFSET,
                            BlockKind('FINALLY'): _FINALLY_POP
                        }[handler['kind']]
                        kwargs['extra_block'] = handler
                        state.fork(pc=tryblk['end'], **kwargs)
                        break
                    else:
                        state.advance_pc()
                        # Must the new PC be a new block?
                        if self._is_implicit_new_block(state):
                            # check if this is a with...as, abort if so
                            self._guard_with_as(state)
                            # else split
                            state.split_new_block()
                            break
                _logger.debug("end state. edges=%s", state.outgoing_edges)
                runner.finished.add(state)
                out_states = state.get_outgoing_states()
                runner.pending.extend(out_states)

        # Complete controlflow
        self._build_cfg(runner.finished)
        # Prune redundant PHI-nodes
        self._prune_phis(runner)
        # Post process
        for state in sorted(runner.finished, key=lambda x: x.pc_initial):
            self.block_infos[state.pc_initial] = si = adapt_state_infos(state)
            _logger.debug("block_infos %s:\n%s", state, si)

    def _build_cfg(self, all_states):
        graph = CFGraph()
        for state in all_states:
            b = state.pc_initial
            graph.add_node(b)
        for state in all_states:
            for edge in state.outgoing_edges:
                graph.add_edge(state.pc_initial, edge.pc, 0)
        graph.set_entry_point(0)
        graph.process()
        self.cfgraph = graph

    def _prune_phis(self, runner):
        # Find phis that are unused in the local block
        _logger.debug("Prune PHIs".center(60, '-'))

        # Compute dataflow for used phis and propagate

        # 1. Get used-phis for each block
        # Map block to used_phis
        def get_used_phis_per_state():
            used_phis = defaultdict(set)
            phi_set = set()
            for state in runner.finished:
                used = set(state._used_regs)
                phis = set(state._phis)
                used_phis[state] |= phis & used
                phi_set |= phis
            return used_phis, phi_set

        # Find use-defs
        def find_use_defs():
            defmap = {}
            phismap = defaultdict(set)
            for state in runner.finished:
                for phi, rhs in state._outgoing_phis.items():
                    if rhs not in phi_set:
                        # Is a definition
                        defmap[phi] = state
                    phismap[phi].add((rhs, state))
            _logger.debug("defmap: %s", _lazy_pformat(defmap))
            _logger.debug("phismap: %s", _lazy_pformat(phismap))
            return defmap, phismap

        def propagate_phi_map(phismap):
            """An iterative dataflow algorithm to find the definition
            (the source) of each PHI node.
            """
            blacklist = defaultdict(set)

            while True:
                changing = False
                for phi, defsites in sorted(list(phismap.items())):
                    for rhs, state in sorted(list(defsites)):
                        if rhs in phi_set:
                            defsites |= phismap[rhs]
                            blacklist[phi].add((rhs, state))
                    to_remove = blacklist[phi]
                    if to_remove & defsites:
                        defsites -= to_remove
                        changing = True

                _logger.debug("changing phismap: %s", _lazy_pformat(phismap))
                if not changing:
                    break

        def apply_changes(used_phis, phismap):
            keep = {}
            for state, used_set in used_phis.items():
                for phi in used_set:
                    keep[phi] = phismap[phi]
            _logger.debug("keep phismap: %s", _lazy_pformat(keep))
            new_out = defaultdict(dict)
            for phi in keep:
                for rhs, state in keep[phi]:
                    new_out[state][phi] = rhs

            _logger.debug("new_out: %s", _lazy_pformat(new_out))
            for state in runner.finished:
                state._outgoing_phis.clear()
                state._outgoing_phis.update(new_out[state])

        used_phis, phi_set = get_used_phis_per_state()
        _logger.debug("Used_phis: %s", _lazy_pformat(used_phis))
        defmap, phismap = find_use_defs()
        propagate_phi_map(phismap)
        apply_changes(used_phis, phismap)
        _logger.debug("DONE Prune PHIs".center(60, '-'))

    def _is_implicit_new_block(self, state):
        inst = state.get_inst()

        if inst.offset in self._bytecode.labels:
            return True
        elif inst.opname in NEW_BLOCKERS:
            return True
        else:
            return False

    def _guard_with_as(self, state):
        """Checks if the next instruction after a SETUP_WITH is something other
        than a POP_TOP, if it is something else it'll be some sort of store
        which is not supported (this corresponds to `with CTXMGR as VAR(S)`)."""
        current_inst = state.get_inst()
        if current_inst.opname == "SETUP_WITH":
            next_op = self._bytecode[current_inst.next].opname
            if next_op != "POP_TOP":
                msg = ("The 'with (context manager) as "
                       "(variable):' construct is not "
                       "supported.")
                raise UnsupportedError(msg)


class TraceRunner(object):
    """Trace runner contains the states for the trace and the opcode dispatch.
    """
    def __init__(self, debug_filename):
        self.debug_filename = debug_filename
        self.pending = deque()
        self.finished = set()

    def get_debug_loc(self, lineno):
        return Loc(self.debug_filename, lineno)

    def dispatch(self, state):
        inst = state.get_inst()
        _logger.debug("dispatch pc=%s, inst=%s", state._pc, inst)
        _logger.debug("stack %s", state._stack)
        fn = getattr(self, "op_{}".format(inst.opname), None)
        if fn is not None:
            fn(state, inst)
        else:
            msg = "Use of unsupported opcode (%s) found" % inst.opname
            raise UnsupportedError(msg, loc=self.get_debug_loc(inst.lineno))

    def op_NOP(self, state, inst):
        state.append(inst)

    def op_FUNC_HEADER(self, state, inst):
        state.append(inst)

    def op_COPY(self, state, inst):
        dst = state.write_register(inst.imm[0])
        src = state.get_register(inst.imm[1])
        state.append(inst, dst=dst, src=src)

    def op_MOVE(self, state, inst):
        dst = state.write_register(inst.imm[0])
        src = state.clear_register(inst.imm[1])
        state.append(inst, dst=dst, src=src)

    def op_CALL_INTRINSIC_1(self, state, inst):
        intrinsic = opcode.intrinsics[inst.imm[0]].name
        value = state._acc  # may be None
        res = state.make_temp()
        state.append(inst, intrinsic=intrinsic, value=value, res=res)
        state.set_acc(res)

    def op_CALL_INTRINSIC_N(self, state, inst):
        res = state.make_temp()
        state.append(inst, res=res)
        state.set_acc(res)

    def op_FORMAT_VALUE(self, state, inst):
        """
        FORMAT_VALUE(flags): flags argument specifies format spec which is
        not supported yet. Currently, we just call str() on the value.
        Pops a value from stack and pushes results back.
        Required for supporting f-strings.
        https://docs.python.org/3/library/dis.html#opcode-FORMAT_VALUE
        """
        if inst.arg != 0:
            msg = "format spec in f-strings not supported yet"
            raise UnsupportedError(msg, loc=self.get_debug_loc(inst.lineno))
        value = state.pop()
        strvar = state.make_temp()
        res = state.make_temp()
        state.append(inst, value=value, res=res, strvar=strvar)
        state.push(res)

    def op_BUILD_STRING(self, state, inst):
        """
        BUILD_STRING(count): Concatenates count strings from the stack and
        pushes the resulting string onto the stack.
        Required for supporting f-strings.
        https://docs.python.org/3/library/dis.html#opcode-BUILD_STRING
        """
        count = inst.arg
        strings = list(reversed([state.pop() for _ in range(count)]))
        # corner case: f""
        if count == 0:
            tmps = [state.make_temp()]
        else:
            tmps = [state.make_temp() for _ in range(count - 1)]
        state.append(inst, strings=strings, tmps=tmps)
        state.push(tmps[-1])

    def op_LOAD_GLOBAL(self, state, inst):
        res = state.make_temp()
        state.append(inst, res=res)
        state.set_acc(res)

    def op_LOAD_DEREF(self, state, inst):
        res = state.make_temp()
        state.append(inst, res=res)
        state.set_acc(res)

    def op_LOAD_CONST(self, state, inst):
        res = state.make_temp("const")
        state.set_acc(res)
        state.append(inst, res=res)

    def op_LOAD_ATTR(self, state, inst):
        item = state.get_register(inst.arg)
        res = state.make_temp()
        state.append(inst, item=item, res=res)
        state.set_acc(res)

    def op_LOAD_FAST(self, state, inst):
        name = state.get_register(inst.arg)
        res = state.make_temp(name)
        state.append(inst, name=name, res=res)
        state.set_acc(res)

    def op_CLEAR_ACC(self, state, inst):
        state.clear_acc()

    def op_CLEAR_FAST(self, state, inst):
        state.clear_register(inst.arg)
        state.append(inst)

    def op_DELETE_FAST(self, state, inst):
        name = state.clear_register(inst.arg)
        state.append(inst, name=name)

    def op_DELETE_ATTR(self, state, inst):
        target = state.acc()
        state.append(inst, target=target)

    def op_STORE_ATTR(self, state, inst):
        target = state.get_register(inst.arg)
        value = state.acc()
        state.append(inst, target=target, value=value)

    def op_STORE_DEREF(self, state, inst):
        value = state.acc()
        state.append(inst, value=value)

    def op_STORE_FAST(self, state, inst):
        name = state.write_register(inst.arg)
        value = state.acc()
        state.clear_acc()
        state.append(inst, value=value, name=name)

    def op_SLICE_1(self, state, inst):
        """
        TOS = TOS1[TOS:]
        """
        tos = state.pop()
        tos1 = state.pop()
        res = state.make_temp()
        slicevar = state.make_temp()
        indexvar = state.make_temp()
        nonevar = state.make_temp()
        state.append(
            inst,
            base=tos1,
            start=tos,
            res=res,
            slicevar=slicevar,
            indexvar=indexvar,
            nonevar=nonevar,
        )
        state.push(res)

    def op_SLICE_2(self, state, inst):
        """
        TOS = TOS1[:TOS]
        """
        tos = state.pop()
        tos1 = state.pop()
        res = state.make_temp()
        slicevar = state.make_temp()
        indexvar = state.make_temp()
        nonevar = state.make_temp()
        state.append(
            inst,
            base=tos1,
            stop=tos,
            res=res,
            slicevar=slicevar,
            indexvar=indexvar,
            nonevar=nonevar,
        )
        state.push(res)

    def op_SLICE_3(self, state, inst):
        """
        TOS = TOS2[TOS1:TOS]
        """
        tos = state.pop()
        tos1 = state.pop()
        tos2 = state.pop()
        res = state.make_temp()
        slicevar = state.make_temp()
        indexvar = state.make_temp()
        state.append(
            inst,
            base=tos2,
            start=tos1,
            stop=tos,
            res=res,
            slicevar=slicevar,
            indexvar=indexvar,
        )
        state.push(res)

    def op_STORE_SLICE_0(self, state, inst):
        """
        TOS[:] = TOS1
        """
        tos = state.pop()
        value = state.pop()
        slicevar = state.make_temp()
        indexvar = state.make_temp()
        nonevar = state.make_temp()
        state.append(
            inst,
            base=tos,
            value=value,
            slicevar=slicevar,
            indexvar=indexvar,
            nonevar=nonevar,
        )

    def op_STORE_SLICE_1(self, state, inst):
        """
        TOS1[TOS:] = TOS2
        """
        tos = state.pop()
        tos1 = state.pop()
        value = state.pop()
        slicevar = state.make_temp()
        indexvar = state.make_temp()
        nonevar = state.make_temp()
        state.append(
            inst,
            base=tos1,
            start=tos,
            slicevar=slicevar,
            value=value,
            indexvar=indexvar,
            nonevar=nonevar,
        )

    def op_STORE_SLICE_2(self, state, inst):
        """
        TOS1[:TOS] = TOS2
        """
        tos = state.pop()
        tos1 = state.pop()
        value = state.pop()
        slicevar = state.make_temp()
        indexvar = state.make_temp()
        nonevar = state.make_temp()
        state.append(
            inst,
            base=tos1,
            stop=tos,
            value=value,
            slicevar=slicevar,
            indexvar=indexvar,
            nonevar=nonevar,
        )

    def op_STORE_SLICE_3(self, state, inst):
        """
        TOS2[TOS1:TOS] = TOS3
        """
        tos = state.pop()
        tos1 = state.pop()
        tos2 = state.pop()
        value = state.pop()
        slicevar = state.make_temp()
        indexvar = state.make_temp()
        state.append(
            inst,
            base=tos2,
            start=tos1,
            stop=tos,
            value=value,
            slicevar=slicevar,
            indexvar=indexvar,
        )

    def op_DELETE_SLICE_0(self, state, inst):
        """
        del TOS[:]
        """
        tos = state.pop()
        slicevar = state.make_temp()
        indexvar = state.make_temp()
        nonevar = state.make_temp()
        state.append(
            inst, base=tos, slicevar=slicevar, indexvar=indexvar,
            nonevar=nonevar,
        )

    def op_DELETE_SLICE_1(self, state, inst):
        """
        del TOS1[TOS:]
        """
        tos = state.pop()
        tos1 = state.pop()
        slicevar = state.make_temp()
        indexvar = state.make_temp()
        nonevar = state.make_temp()
        state.append(
            inst,
            base=tos1,
            start=tos,
            slicevar=slicevar,
            indexvar=indexvar,
            nonevar=nonevar,
        )

    def op_DELETE_SLICE_2(self, state, inst):
        """
        del TOS1[:TOS]
        """
        tos = state.pop()
        tos1 = state.pop()
        slicevar = state.make_temp()
        indexvar = state.make_temp()
        nonevar = state.make_temp()
        state.append(
            inst,
            base=tos1,
            stop=tos,
            slicevar=slicevar,
            indexvar=indexvar,
            nonevar=nonevar,
        )

    def op_DELETE_SLICE_3(self, state, inst):
        """
        del TOS2[TOS1:TOS]
        """
        tos = state.pop()
        tos1 = state.pop()
        tos2 = state.pop()
        slicevar = state.make_temp()
        indexvar = state.make_temp()
        state.append(
            inst, base=tos2, start=tos1, stop=tos, slicevar=slicevar,
            indexvar=indexvar
        )

    def op_BUILD_SLICE(self, state, inst):
        """
        slice(TOS1, TOS) or slice(TOS2, TOS1, TOS)
        """
        base = inst.arg
        start, stop, step = [state.get_register(base + i) for i in range(3)]
        slicevar = state.make_temp()
        res = state.make_temp()
        state.append(
            inst, start=start, stop=stop, step=step, res=res, slicevar=slicevar
        )
        state.set_acc(res)

    def _op_POP_JUMP_IF(self, state, inst):
        pred = state.acc()
        state.clear_acc()
        state.append(inst, pred=pred)

        target_inst = inst.get_jump_target()
        next_inst = inst.next
        # if the next inst and the jump target are the same location, issue one
        # fork else issue a fork for the next and the target.
        state.fork(pc=next_inst)
        if target_inst != next_inst:
            state.fork(pc=target_inst)

    op_POP_JUMP_IF_TRUE = _op_POP_JUMP_IF
    op_POP_JUMP_IF_FALSE = _op_POP_JUMP_IF

    def _op_JUMP_IF(self, state, inst):
        pred = state.acc()
        state.append(inst, pred=pred)
        state.fork(pc=inst.next)
        state.fork(pc=inst.get_jump_target())

    op_JUMP_IF_FALSE = _op_JUMP_IF
    op_JUMP_IF_TRUE = _op_JUMP_IF

    def op_JUMP(self, state, inst):
        state.append(inst)
        state.fork(pc=inst.get_jump_target())

    def op_RETURN_VALUE(self, state, inst):
        state.append(inst, retval=state.acc(), castval=state.make_temp())
        state.terminate()

    def op_YIELD_VALUE(self, state, inst):
        val = state.acc()
        res = state.make_temp()
        state.append(inst, value=val, res=res)
        state.set_acc(res)

    def op_RAISE(self, state, inst):
        in_exc_block = any([
            state.get_top_block("EXCEPT") is not None,
            state.get_top_block("FINALLY") is not None
        ])
        if state._acc is None:
            exc = None
            if in_exc_block:
                raise UnsupportedError(
                    "The re-raising of an exception is not yet supported.",
                    loc=self.get_debug_loc(inst.lineno),
                )
        else:
            exc = state.acc()
        state.append(inst, exc=exc)
        state.terminate()

    def op_BEGIN_FINALLY(self, state, inst):
        temps = []
        for i in range(_EXCEPT_STACK_OFFSET):
            tmp = state.make_temp()
            temps.append(tmp)
            state.push(tmp)
        state.append(inst, temps=temps)

    def op_END_FINALLY(self, state, inst):
        blk = state.pop_block()
        state.reset_stack(blk['entry_stack'])

    def op_POP_FINALLY(self, state, inst):
        # we don't emulate the exact stack behavior
        if inst.arg != 0:
            msg = ('Unsupported use of a bytecode related to try..finally'
                   ' or a with-context')
            raise UnsupportedError(msg, loc=self.get_debug_loc(inst.lineno))

    def op_CALL_FINALLY(self, state, inst):
        pass

    def op_WITH_CLEANUP_START(self, state, inst):
        # we don't emulate the exact stack behavior
        state.append(inst)

    def op_WITH_CLEANUP_FINISH(self, state, inst):
        # we don't emulate the exact stack behavior
        state.append(inst)

    def op_SETUP_WITH(self, state, inst):
        assert False
        cm = state.pop()    # the context-manager

        yielded = state.make_temp()
        exitfn = state.make_temp(prefix='setup_with_exitfn')
        state.append(inst, contextmanager=cm, exitfn=exitfn)

        # py39 doesn't have with-finally
        if PYVERSION < (3, 9):
            state.push_block(
                state.make_block(
                    kind='WITH_FINALLY',
                    end=inst.get_jump_target(),
                )
            )

        state.push(exitfn)
        state.push(yielded)

        state.push_block(
            state.make_block(
                kind='WITH',
                end=inst.get_jump_target(),
            )
        )
        # Forces a new block
        state.fork(pc=inst.next)

    def _setup_try(self, kind, state, next, end):
        handler_block = state.make_block(
            kind=kind,
            end=None,
            reset_stack=False,
        )
        # Forces a new block
        # Fork to the body of the finally
        state.fork(
            pc=next,
            extra_block=state.make_block(
                kind='TRY',
                end=end,
                reset_stack=False,
                handler=handler_block,
            )
        )

    def op_SETUP_EXCEPT(self, state, inst):
        # Opcode removed since py3.8
        state.append(inst)
        self._setup_try(
            'EXCEPT', state, next=inst.next, end=inst.get_jump_target(),
        )

    def op_SETUP_FINALLY(self, state, inst):
        state.append(inst)
        self._setup_try(
            'FINALLY', state, next=inst.next, end=inst.get_jump_target(),
        )

    def op_POP_EXCEPT(self, state, inst):
        blk = state.pop_block()
        if blk['kind'] not in {BlockKind('EXCEPT'), BlockKind('FINALLY')}:
            raise UnsupportedError(
                "POP_EXCEPT got an unexpected block: {}".format(blk['kind']),
                loc=self.get_debug_loc(inst.lineno),
            )
        state.pop()
        state.pop()
        state.pop()
        # Forces a new block
        state.fork(pc=inst.next)

    def op_BINARY_SUBSCR(self, state, inst):
        index = state.acc()
        target = state.get_register(inst.arg)
        res = state.make_temp()
        state.append(inst, index=index, target=target, res=res)
        state.set_acc(res)

    def op_STORE_SUBSCR(self, state, inst):
        index = state.get_register(inst.imm[1])
        target = state.get_register(inst.imm[0])
        value = state.acc()
        state.append(inst, target=target, index=index, value=value)

    def op_DELETE_SUBSCR(self, state, inst):
        index = state.acc()
        target = state.get_register(inst.imm[0])
        state.append(inst, target=target, index=index)

    def op_CALL_FUNCTION(self, state, inst, is_method=False):
        base, narg = inst.imm
        if is_method:
            base += 1
            narg -= 1
        nkwarg = narg >> 8
        narg = narg & 0xFF
        args = [state.clear_register(base + i) for i in range(narg)]
        func = state.clear_register(base - 1)
        if nkwarg > 0:
            kwnames = state.clear_register(base - 5)
            kwargs = [state.clear_register(base - 5 - nkwarg + i) for i in range(nkwarg)]
        else:
            kwnames = None
            kwargs = ()
        res = state.make_temp()
        state.append(inst, func=func, args=args, kwnames=kwnames, kwargs=kwargs, res=res)
        state.set_acc(res)

    def op_CALL_FUNCTION_EX(self, state, inst):
        base = inst.imm[0]
        vararg = state.clear_register(base - 6)
        # kwarg = state.clear_register(base - 5)
        # if kwarg is not None:
        #     errmsg = "CALL_FUNCTION_EX with **kwargs not supported"
        #     raise UnsupportedError(errmsg)
        func = state.clear_register(base - 1)
        res = state.make_temp()
        state.append(inst, func=func, vararg=vararg, res=res)
        state.set_acc(res)

    def op_UNPACK(self, state, inst):
        base, count, after = inst.imm
        iterable = state.acc()
        stores = [state.write_register(base + i) for i in reversed(range(count))]
        tupleobj = state.make_temp()
        state.append(inst, iterable=iterable, stores=stores, tupleobj=tupleobj)
        state.clear_acc()

    def op_BUILD_TUPLE(self, state, inst):
        base, count = inst.imm
        items = [state.clear_register(base + i) for i in range(count)]
        tup = state.make_temp()
        state.append(inst, items=items, res=tup)
        state.set_acc(tup)

    def op_LIST_TO_TUPLE(self, state, inst):
        # "Pops a list from the stack and pushes a tuple containing the same
        #  values."
        tos = state.pop()
        res = state.make_temp() # new tuple var
        state.append(inst, const_list=tos, res=res)
        state.push(res)

    def op_BUILD_LIST(self, state, inst):
        base, count = inst.imm
        items = [state.clear_register(base + i) for i in range(count)]
        lst = state.make_temp()
        state.append(inst, items=items, res=lst)
        state.set_acc(lst)

    def op_LIST_APPEND(self, state, inst):
        value = state.acc()
        target = state.get_register(inst.arg)
        appendvar = state.make_temp()
        res = state.make_temp()
        state.append(inst, target=target, value=value, appendvar=appendvar,
                     res=res)

    def op_LIST_EXTEND(self, state, inst):
        value = state.acc()
        target = state.get_register(inst.arg)
        extendvar = state.make_temp()
        res = state.make_temp()
        state.append(inst, target=target, value=value, extendvar=extendvar,
                     res=res)
        state.clear_acc()

    def op_BUILD_MAP(self, state, inst):
        dct = state.make_temp()
        state.append(inst, res=dct)
        state.set_acc(dct)

    def op_BUILD_SET(self, state, inst):
        base, count = inst.imm
        items = [state.get_register(base + i) for i in range(count)]
        lst = state.make_temp()
        state.append(inst, items=items, res=lst)
        state.set_acc(lst)

    def op_SET_UPDATE(self, state, inst):
        value = state.acc()
        target = state.get_register(inst.arg)
        updatevar = state.make_temp()
        res = state.make_temp()
        state.append(inst, target=target, value=value, updatevar=updatevar,
                     res=res)
        state.clear_acc()

    def op_GET_ITER(self, state, inst):
        value = state.acc()
        res = state.write_register(inst.arg)
        state.append(inst, value=value, res=res)
        state.clear_acc()

    def op_FOR_ITER(self, state, inst):
        iterator = state.get_register(inst.imm[0])
        pair = state.make_temp()
        indval = state.make_temp()
        pred = state.make_temp()
        state.append(inst, iterator=iterator, pair=pair, indval=indval,
                     pred=pred)
        state.set_acc(indval)
        loop = inst.get_jump_target()
        state.fork(pc=loop)
        state.clear_register(inst.imm[0])
        state.fork(pc=inst.next, clear_acc=True)

    def _unaryop(self, state, inst):
        val = state.acc()
        res = state.make_temp()
        state.append(inst, value=val, res=res)
        state.set_acc(res)

    op_UNARY_NEGATIVE = _unaryop
    op_UNARY_POSITIVE = _unaryop
    op_UNARY_NOT = _unaryop
    op_UNARY_NOT_FAST = _unaryop
    op_UNARY_INVERT = _unaryop

    def _binaryop(self, state, inst):
        rhs = state.acc()
        lhs = state.get_register(inst.arg)
        res = state.make_temp()
        state.append(inst, lhs=lhs, rhs=rhs, res=res)
        state.set_acc(res)

    def op_COMPARE_OP(self, state, inst):
        rhs = state.acc()
        lhs = state.get_register(inst.imm[1])
        res = state.make_temp()
        state.append(inst, lhs=lhs, rhs=rhs, res=res)
        state.set_acc(res)

    op_IS_OP = _binaryop
    op_CONTAINS_OP = _binaryop

    op_INPLACE_ADD = _binaryop
    op_INPLACE_SUBTRACT = _binaryop
    op_INPLACE_MULTIPLY = _binaryop
    op_INPLACE_DIVIDE = _binaryop
    op_INPLACE_TRUE_DIVIDE = _binaryop
    op_INPLACE_FLOOR_DIVIDE = _binaryop
    op_INPLACE_MODULO = _binaryop
    op_INPLACE_POWER = _binaryop
    op_INPLACE_MATRIX_MULTIPLY = _binaryop

    op_INPLACE_LSHIFT = _binaryop
    op_INPLACE_RSHIFT = _binaryop
    op_INPLACE_AND = _binaryop
    op_INPLACE_OR = _binaryop
    op_INPLACE_XOR = _binaryop

    op_BINARY_ADD = _binaryop
    op_BINARY_SUBTRACT = _binaryop
    op_BINARY_MULTIPLY = _binaryop
    op_BINARY_DIVIDE = _binaryop
    op_BINARY_TRUE_DIVIDE = _binaryop
    op_BINARY_FLOOR_DIVIDE = _binaryop
    op_BINARY_MODULO = _binaryop
    op_BINARY_POWER = _binaryop
    op_BINARY_MATRIX_MULTIPLY = _binaryop

    op_BINARY_LSHIFT = _binaryop
    op_BINARY_RSHIFT = _binaryop
    op_BINARY_AND = _binaryop
    op_BINARY_OR = _binaryop
    op_BINARY_XOR = _binaryop

    def op_MAKE_FUNCTION(self, state, inst):
        res = state.make_temp()
        code = state.code_consts[inst.arg]
        name = code.co_consts[1] if len(code.co_consts) > 0 else code.co_name
        annotations = None
        freevars = tuple([state.get_register(pair[0]) for pair in code.co_free2reg])
        defaults = None
        closure = None
        kwdefaults = None

        if code.co_ndefaultargs > 0:
            defaults = freevars[:code.co_ndefaultargs]
        if len(code.co_freevars) > code.co_ndefaultargs:
            closure = freevars[code.co_ndefaultargs:]

        state.append(
            inst,
            name=name,
            code=code,
            closure=closure,
            kwdefaults=kwdefaults,
            defaults=defaults,
            res=res,
        )
        state.set_acc(res)

    def op_MAKE_CLOSURE(self, state, inst):
        self.op_MAKE_FUNCTION(state, inst, MAKE_CLOSURE=True)

    def op_LOAD_CLOSURE(self, state, inst):
        res = state.make_temp()
        state.append(inst, res=res)
        state.push(res)

    def op_LOAD_ASSERTION_ERROR(self, state, inst):
        res = state.make_temp("assertion_error")
        state.append(inst, res=res)
        state.push(res)

    def op_JUMP_IF_NOT_EXC_MATCH(self, state, inst):
        # Tests whether the second value on the stack is an exception matching
        # TOS, and jumps if it is not. Pops two values from the stack.
        pred = state.make_temp("predicate")
        tos = state.pop()
        tos1 = state.pop()
        state.append(inst, pred=pred, tos=tos, tos1=tos1)
        state.fork(pc=inst.next)
        state.fork(pc=inst.get_jump_target())

    def op_RERAISE(self, state, inst):
        # This isn't handled, but the state is set up anyway
        exc = state.pop()
        state.append(inst, exc=exc)
        state.terminate()

    # NOTE: Please see notes in `interpreter.py` surrounding the implementation
    # of LOAD_METHOD and CALL_METHOD.

    def op_LOAD_METHOD(self, state, inst):
        # NOTE: shifted by one compared to LOAD_ATTR
        item = state.acc()
        res = state.write_register(inst.imm[0] + 1)
        state.append(inst, item=item, res=res)
        state.set_acc(res)

    def op_CALL_METHOD(self, state, inst):
        self.op_CALL_FUNCTION(state, inst, is_method=True)


@total_ordering
class State(object):
    """State of the trace
    """
    def __init__(self, bytecode, pc, acc=None, temporaries=None):
        """
        Parameters
        ----------
        bytecode : numba.bytecode.ByteCode
            function bytecode
        pc : int
            program counter
        nstack : int
            stackdepth at entry
        blockstack : Sequence[Dict]
            A sequence of dictionary denoting entries on the blockstack.
        """
        self._bytecode = bytecode
        self._pc_initial = pc
        self._pc = pc
        self._acc = acc
        self._stack = []
        self._blockstack_initial = ()
        self._temporaries = {}
        self._temp_registers = []
        self._insts = []
        self._outedges = []
        self._terminated = False
        self._phis = {}
        self._outgoing_phis = UniqueDict()
        self._used_regs = set()
        if acc is not None:
            phi = self.make_temp("phi")
            self._phis[phi] = 'acc'
            self.set_acc(phi)
        if temporaries is not None:
            for k, v in temporaries.items():
                phi = self.make_temp("phi")
                self._phis[phi] = k
                self._temporaries[k] = phi

    def __repr__(self):
        return "State(pc_initial={} nstack_initial={})".format(
            self._pc_initial, None
        )

    def get_identity(self):
        return (self._pc_initial, 0)

    def __hash__(self):
        return hash(self.get_identity())

    def __lt__(self, other):
        return self.get_identity() < other.get_identity()

    def __eq__(self, other):
        return self.get_identity() == other.get_identity()

    @property
    def pc_initial(self):
        """The starting bytecode offset of this State.
        The PC given to the constructor.
        """
        return self._pc_initial

    @property
    def instructions(self):
        """The list of instructions information as a 2-tuple of
        ``(pc : int, register_map : Dict)``
        """
        return self._insts

    @property
    def outgoing_edges(self):
        """The list of outgoing edges.

        Returns
        -------
        edges : List[State]
        """
        return self._outedges

    @property
    def outgoing_phis(self):
        """The dictionary of outgoing phi nodes.

        The keys are the name of the PHI nodes.
        The values are the outgoing states.
        """
        return self._outgoing_phis

    @property
    def blockstack_initial(self):
        """A copy of the initial state of the blockstack
        """
        return self._blockstack_initial

    @property
    def stack_depth(self):
        """The current size of the stack

        Returns
        -------
        res : int
        """
        return len(self._stack)

    def find_initial_try_block(self):
        """Find the initial *try* block.
        """
        for blk in reversed(self._blockstack_initial):
            if blk['kind'] == BlockKind('TRY'):
                return blk

    def has_terminated(self):
        return self._terminated

    def get_inst(self):
        return self._bytecode[self._pc]

    def advance_pc(self):
        inst = self.get_inst()
        self._pc = inst.next

    def make_temp(self, prefix=""):
        if not prefix:
            name = "${prefix}{offset}{opname}.{tempct}".format(
                prefix=prefix,
                offset=self._pc,
                opname=self.get_inst().opname.lower(),
                tempct=len(self._temp_registers),
            )
        else:
            name = "${prefix}{offset}.{tempct}".format(
                prefix=prefix,
                offset=self._pc,
                tempct=len(self._temp_registers),
            )

        self._temp_registers.append(name)
        return name

    def append(self, inst, **kwargs):
        """Append new inst"""
        self._insts.append((inst.offset, kwargs))
        self._used_regs |= set(_flatten_inst_regs(kwargs.values()))

    def set_acc(self, item):
        self._acc = item

    def acc(self):
        assert self._acc is not None
        return self._acc

    def clear_acc(self):
        self._acc = None

    def push_block(self, synblk):
        """Push a block to blockstack
        """
        assert 'stack_depth' in synblk
        self._blockstack.append(synblk)

    def reset_stack(self, depth):
        """Reset the stack to the given stack depth.
        Returning the popped items.
        """
        self._stack, popped = self._stack[:depth], self._stack[depth:]
        return popped

    def make_block(self, kind, end, reset_stack=True, handler=None):
        """Make a new block
        """
        d = {
            'kind': BlockKind(kind),
            'end': end,
            'entry_stack': len(self._stack),
        }
        if reset_stack:
            d['stack_depth'] = len(self._stack)
        else:
            d['stack_depth'] = None
        d['handler'] = handler
        return d

    def pop_block(self):
        """Pop a block and unwind the stack
        """
        b = self._blockstack.pop()
        self.reset_stack(b['stack_depth'])
        return b

    def pop_block_and_above(self, blk):
        """Find *blk* in the blockstack and remove it and all blocks above it
        from the stack.
        """
        idx = self._blockstack.index(blk)
        assert 0 <= idx < len(self._blockstack)
        self._blockstack = self._blockstack[:idx]

    def get_top_block(self, kind):
        """Find the first block that matches *kind*
        """
        return None
        # kind = BlockKind(kind)
        # for bs in reversed(self._blockstack):
        #     if bs['kind'] == kind:
        #         return bs

    def has_active_try(self):
        """Returns a boolean indicating if the top-block is a *try* block
        """
        return self.get_top_block('TRY') is not None

    @property
    def code_consts(self):
        return self._bytecode.co_consts

    def write_temporary(self, idx):
        name = self.make_temp()
        self._temporaries[idx] = name
        return name

    def clear_temporary(self, idx):
        return self._temporaries.pop(idx)

    def read_temporary(self, idx):
        return self._temporaries[idx]

    def write_register(self, idx):
        if idx >= self._bytecode.num_locals:
            return self.write_temporary(idx - self._bytecode.num_locals)
        return self.get_register(idx)

    def clear_register(self, idx):
        if idx >= self._bytecode.num_locals:
            return self.clear_temporary(idx - self._bytecode.num_locals)
        return self.get_register(idx)

    def get_register(self, idx):
        if idx >= self._bytecode.num_locals:
            return self.read_temporary(idx - self._bytecode.num_locals)
        return self._bytecode.co_varnames[idx]

    def terminate(self):
        """Mark block as terminated
        """
        self._terminated = True

    def fork(self, pc, clear_acc=False):
        """Fork the state
        """
        # Handle changes on the stack
        acc = self._acc if not clear_acc else None
        temporaries = dict(self._temporaries)
        # assert npop == 0 and npush == 0
        # if npop:
        #     assert 0 <= npop <= len(self._stack)
        #     nstack = len(self._stack) - npop
        #     stack = stack[:nstack]
        # if npush:
        #     assert 0 <= npush
        #     for i in range(npush):
        #         stack.append(self.make_temp())
        # # Handle changes on the blockstack
        # blockstack = list(self._blockstack)
        # if extra_block:
        #     blockstack.append(extra_block)
        self._outedges.append(Edge(pc=pc, acc=acc, temporaries=temporaries))
        self.terminate()

    def split_new_block(self):
        """Split the state
        """
        self.fork(pc=self._pc)

    def get_outgoing_states(self):
        """Get states for each outgoing edges
        """
        # Should only call once
        assert not self._outgoing_phis
        ret = []
        for edge in self._outedges:
            state = State(bytecode=self._bytecode, pc=edge.pc,
                          acc=edge.acc, temporaries=edge.temporaries)
            ret.append(state)
            # Map outgoing_phis
            for phi, i in state._phis.items():
                if i == 'acc':
                    self._outgoing_phis[phi] = edge.acc
                else:
                    self._outgoing_phis[phi] = edge.temporaries[i]
        return ret

    def get_outgoing_edgepushed(self):
        """
        Returns
        -------
        Dict[int, int]
            where keys are the PC
            values are the edge-pushed stack values
        """

        return {edge.pc: edge.acc
                for edge in self._outedges}


Edge = namedtuple("Edge", ["pc", "acc", "temporaries"])
# Edge = namedtuple("Edge", ["pc", "stack", "blockstack", "npush"])


class AdaptDFA(object):
    """Adapt Flow to the old DFA class expected by Interpreter
    """
    def __init__(self, flow):
        self._flow = flow

    @property
    def infos(self):
        return self._flow.block_infos


AdaptBlockInfo = namedtuple(
    "AdaptBlockInfo",
    ["insts", "outgoing_phis", "blockstack", "active_try_block",
     "outgoing_edgepushed"],
)


def adapt_state_infos(state):
    return AdaptBlockInfo(
        insts=tuple(state.instructions),
        outgoing_phis=state.outgoing_phis,
        blockstack=state.blockstack_initial,
        active_try_block=state.find_initial_try_block(),
        outgoing_edgepushed=state.get_outgoing_edgepushed(),
    )


def _flatten_inst_regs(iterable):
    """Flatten an iterable of registers used in an instruction
    """
    for item in iterable:
        if isinstance(item, str):
            yield item
        elif isinstance(item, (tuple, list)):
            for x in _flatten_inst_regs(item):
                yield x


class AdaptCFA(object):
    """Adapt Flow to the old CFA class expected by Interpreter
    """
    def __init__(self, flow):
        self._flow = flow
        self._blocks = {}
        for offset, blockinfo in flow.block_infos.items():
            self._blocks[offset] = AdaptCFBlock(blockinfo, offset)
        backbone = self._flow.cfgraph.backbone()

        graph = flow.cfgraph
        # Find backbone
        backbone = graph.backbone()
        # Filter out in loop blocks (Assuming no other cyclic control blocks)
        # This is to unavoid variables defined in loops being considered as
        # function scope.
        inloopblocks = set()
        for b in self.blocks.keys():
            if graph.in_loops(b):
                inloopblocks.add(b)
        self._backbone = backbone - inloopblocks

    @property
    def graph(self):
        return self._flow.cfgraph

    @property
    def backbone(self):
        return self._backbone

    @property
    def blocks(self):
        return self._blocks

    def iterliveblocks(self):
        succs = self.graph._succs
        seen = set()
        order = []

        def _dfs_rec(node):
            if node not in seen:
                order.append(node)
                seen.add(node)
                for dest in sorted(succs[node]):
                    _dfs_rec(dest)
        _dfs_rec(self.graph.entry_point())
        for b in order:
            yield self.blocks[b]

    def dump(self):
        self._flow.cfgraph.dump()


class AdaptCFBlock(object):
    def __init__(self, blockinfo, offset):
        self.offset = offset
        self.body = tuple(i for i, _ in blockinfo.insts)
