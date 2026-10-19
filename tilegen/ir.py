"""TileGen intermediate representation (TACO-style).

The IR is a small tree of dataclasses:

    Format      -- level structure of a tensor ("dense" | "compressed")
    TensorDecl  -- a named tensor with an order, format, dtype and shape
    IndexExpr   -- Access / Mul / Add / Const / Reduce
    Stmt        -- Assign(lhs_access, rhs_expr)
    Schedule    -- ordered list of transforms (Tile/Split/...)
    Kernel      -- name, stmts, schedule, formats; the unit passes operate on

``Kernel.validate()`` performs structural well-formedness checks and is the
gate every pass and the code generator rely on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Index",
    "Format",
    "DENSE1",
    "DENSE2",
    "CSR",
    "CSC",
    "COO",
    "ELL",
    "BSR",
    "SCALAR",
    "FORMAT_REGISTRY",
    "TensorDecl",
    "Expr",
    "Access",
    "Mul",
    "Add",
    "Const",
    "Reduce",
    "Stmt",
    "Assign",
    "Transform",
    "Tile",
    "Split",
    "Parallelize",
    "Unroll",
    "Fuse",
    "ThreadsPerRow",
    "Vectorize",
    "Schedule",
    "Kernel",
]


# --------------------------------------------------------------------------- #
# Index
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Index:
    """A loop / iteration variable such as ``i`` or ``j``."""

    name: str

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.name


# --------------------------------------------------------------------------- #
# Format
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Format:
    """Storage format described by its per-level kinds.

    ``levels`` is a list whose entries are ``"dense"`` or ``"compressed"``.
    ``ell_width`` and ``block_size`` parameterize ELL and BSR respectively.
    """

    name: str
    levels: Tuple[str, ...]
    ell_width: Optional[int] = None
    block_size: Optional[int] = None

    @property
    def order(self) -> int:
        return len(self.levels)

    def is_sparse(self) -> bool:
        return "compressed" in self.levels

    def compressed_positions(self) -> List[int]:
        return [i for i, lv in enumerate(self.levels) if lv == "compressed"]


DENSE1 = Format("dense1", ("dense",))
DENSE2 = Format("dense2", ("dense", "dense"))
CSR = Format("csr", ("dense", "compressed"))
CSC = Format("csc", ("compressed", "dense"))
COO = Format("coo", ("compressed", "compressed"))
ELL = Format("ell", ("dense", "compressed"), ell_width=8)
BSR = Format("bsr", ("dense", "compressed"), block_size=4)
SCALAR = Format("scalar", ())

FORMAT_REGISTRY: Dict[str, Format] = {
    "dense1": DENSE1,
    "dense2": DENSE2,
    "csr": CSR,
    "csc": CSC,
    "coo": COO,
    "ell": ELL,
    "bsr": BSR,
    "scalar": SCALAR,
}


# --------------------------------------------------------------------------- #
# Tensor declaration
# --------------------------------------------------------------------------- #
@dataclass
class TensorDecl:
    """A tensor used by a kernel.

    ``shape`` entries are symbolic dimension names (strings) that the code
    generator turns into ABI arguments, e.g. ``("M", "N")``.
    """

    name: str
    order: int
    format: Format
    dtype: str = "f64"
    shape: Tuple[Any, ...] = ()

    def is_scalar(self) -> bool:
        return self.order == 0

    def __hash__(self) -> int:
        return hash((self.name, self.order, self.dtype))


# --------------------------------------------------------------------------- #
# Index expressions
# --------------------------------------------------------------------------- #
class Expr:
    """Base class for all index expressions."""

    # Operator overloading so the frontend reads like math.
    def __mul__(self, other: "Expr") -> "Mul":
        return Mul(self, _as_expr(other))

    def __rmul__(self, other: "Expr") -> "Mul":
        return Mul(_as_expr(other), self)

    def __add__(self, other: "Expr") -> "Add":
        return Add(self, _as_expr(other))

    def __radd__(self, other: "Expr") -> "Add":
        return Add(_as_expr(other), self)

    def __sub__(self, other: "Expr") -> "Add":
        return Add(self, Mul(Const(-1.0), _as_expr(other)))

    def walk(self):
        """Yield this node and all descendants (pre-order)."""
        yield self
        for child in self.children():
            yield from child.walk()

    def children(self) -> List["Expr"]:
        return []


def _as_expr(value: Any) -> Expr:
    if isinstance(value, Expr):
        return value
    if isinstance(value, TensorDecl):
        return Access(value, [])
    if isinstance(value, (int, float)):
        return Const(float(value))
    raise TypeError("cannot convert %r to an expression" % (value,))


@dataclass(eq=False)
class Access(Expr):
    tensor: TensorDecl
    indices: List[Index]

    def children(self) -> List[Expr]:
        return []

    def bound_indices(self) -> List[Index]:
        return list(self.indices)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        inner = ", ".join(idx.name for idx in self.indices)
        return "%s[%s]" % (self.tensor.name, inner)


@dataclass(eq=False)
class Mul(Expr):
    left: Expr
    right: Expr

    def children(self) -> List[Expr]:
        return [self.left, self.right]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "(%r * %r)" % (self.left, self.right)


@dataclass(eq=False)
class Add(Expr):
    left: Expr
    right: Expr

    def children(self) -> List[Expr]:
        return [self.left, self.right]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "(%r + %r)" % (self.left, self.right)


@dataclass(eq=False)
class Const(Expr):
    value: float

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return repr(self.value)


@dataclass(eq=False)
class Reduce(Expr):
    index: Index
    body: Expr

    def children(self) -> List[Expr]:
        return [self.body]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "sum(%s, %r)" % (self.index.name, self.body)


# --------------------------------------------------------------------------- #
# Statements
# --------------------------------------------------------------------------- #
@dataclass(eq=False)
class Stmt:
    lhs: Access
    rhs: Expr


def Assign(lhs: Access, rhs: Expr) -> Stmt:
    """Construct an assignment statement (documented sugar; ``==`` also works)."""
    return Stmt(lhs=lhs, rhs=rhs)


# ``y[i] == expr`` builds an assignment.  Access has eq=False so this override
# does not disturb structural comparisons elsewhere.
def _access_eq(self: Access, other: Any) -> Stmt:  # type: ignore[override]
    return Stmt(lhs=self, rhs=_as_expr(other))


Access.__eq__ = _access_eq  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# Schedule transforms
# --------------------------------------------------------------------------- #
@dataclass
class Transform:
    kind: ClassVar[str] = "transform"


@dataclass
class Tile(Transform):
    index: Index = None  # type: ignore[assignment]
    size: int = 0
    kind: ClassVar[str] = "tile"


@dataclass
class Split(Transform):
    index: Index = None  # type: ignore[assignment]
    factor: int = 0
    kind: ClassVar[str] = "split"


@dataclass
class Parallelize(Transform):
    index: Index = None  # type: ignore[assignment]
    unit: str = "thread"
    kind: ClassVar[str] = "parallelize"


@dataclass
class Unroll(Transform):
    index: Index = None  # type: ignore[assignment]
    factor: int = 0
    kind: ClassVar[str] = "unroll"


@dataclass
class Fuse(Transform):
    index_a: Index = None  # type: ignore[assignment]
    index_b: Index = None  # type: ignore[assignment]
    kind: ClassVar[str] = "fuse"


@dataclass
class ThreadsPerRow(Transform):
    k: int = 1
    kind: ClassVar[str] = "threads_per_row"


@dataclass
class Vectorize(Transform):
    width: int = 1
    kind: ClassVar[str] = "vectorize"


@dataclass
class Schedule:
    transforms: List[Transform] = field(default_factory=list)

    def append(self, t: Transform) -> "Schedule":
        self.transforms.append(t)
        return self

    def __iter__(self):
        return iter(self.transforms)

    def __len__(self):
        return len(self.transforms)


# --------------------------------------------------------------------------- #
# Kernel
# --------------------------------------------------------------------------- #
@dataclass(eq=False)
class Kernel:
    name: str
    stmts: List[Stmt]
    schedule: Schedule = field(default_factory=Schedule)
    formats: List[Format] = field(default_factory=list)
    dtypes: List[str] = field(default_factory=list)
    pass_log: List[Dict[str, Any]] = field(default_factory=list)

    # -- convenience ------------------------------------------------------- #
    @property
    def stmt(self) -> Stmt:
        """The first statement (single-statement kernels)."""
        return self.stmts[0]

    def tensors(self) -> List[TensorDecl]:
        seen: List[TensorDecl] = []
        names = set()
        for st in self.stmts:
            for node in list(st.rhs.walk()) + [st.lhs]:
                if isinstance(node, Access) and node.tensor.name not in names:
                    names.add(node.tensor.name)
                    seen.append(node.tensor)
        return seen

    def free_indices(self) -> List[Index]:
        """Ordered indices that define the iteration space (not reduce-bound)."""
        out: List[Index] = []
        seen = set()

        def add(idx: Index):
            if idx.name not in seen:
                seen.add(idx.name)
                out.append(idx)

        for st in self.stmts:
            for idx in st.lhs.indices:
                add(idx)
        for st in self.stmts:
            for idx in self._free_of(st.rhs, set()):
                add(idx)
        return out

    def _free_of(self, expr: Expr, bound: set) -> List[Index]:
        out: List[Index] = []
        if isinstance(expr, Reduce):
            inner = self._free_of(expr.body, bound | {expr.index.name})
            return inner
        if isinstance(expr, Access):
            for idx in expr.indices:
                if idx.name not in bound:
                    out.append(idx)
            return out
        for child in expr.children():
            out.extend(self._free_of(child, bound))
        return out

    def reduce_indices(self) -> List[Index]:
        out: List[Index] = []
        for st in self.stmts:
            for node in st.rhs.walk():
                if isinstance(node, Reduce):
                    out.append(node.index)
        return out

    def all_indices(self) -> List[Index]:
        out: List[Index] = []
        seen = set()
        for idx in self.free_indices() + self.reduce_indices():
            if idx.name not in seen:
                seen.add(idx.name)
                out.append(idx)
        return out

    def reduce_node(self) -> Optional[Reduce]:
        for st in self.stmts:
            for node in st.rhs.walk():
                if isinstance(node, Reduce):
                    return node
        return None

    def sparse_tensor(self) -> Optional[TensorDecl]:
        for t in self.tensors():
            if t.format.is_sparse():
                return t
        return None

    # -- validation -------------------------------------------------------- #
    def validate(self) -> bool:
        """Raise ValueError describing the first structural problem found."""
        if not self.stmts:
            raise ValueError("kernel %r has no statements" % self.name)

        # Formats must be self-consistent.
        for fmt in self.formats:
            if len(fmt.levels) != fmt.order:
                raise ValueError(
                    "format %r: %d levels for order %d" % (fmt.name, len(fmt.levels), fmt.order)
                )
            for lv in fmt.levels:
                if lv not in ("dense", "compressed"):
                    raise ValueError("format %r has invalid level %r" % (fmt.name, lv))

        # Access ranks must equal tensor orders; format order must match.
        for st in self.stmts:
            for node in list(st.rhs.walk()) + [st.lhs]:
                if isinstance(node, Access):
                    t = node.tensor
                    if len(node.indices) != t.order:
                        raise ValueError(
                            "access %s has %d indices but order is %d"
                            % (t.name, len(node.indices), t.order)
                        )
                    if t.format.order != t.order:
                        raise ValueError(
                            "tensor %s: format %r order %d != declared order %d"
                            % (t.name, t.format.name, t.format.order, t.order)
                        )

            # (a) every lhs index must be bound by the rhs iteration space.
            rhs_free = {idx.name for idx in self._free_of(st.rhs, set())}
            for idx in st.lhs.indices:
                if idx.name not in rhs_free:
                    raise ValueError(
                        "index %r appears in the lhs but is not bound by the rhs"
                        % idx.name
                    )

            # reduce indices must actually occur in the reduced body.
            for node in st.rhs.walk():
                if isinstance(node, Reduce):
                    body_idx = {i.name for i in self._free_of(node.body, set())}
                    body_idx |= {
                        i.name for n in node.body.walk()
                        if isinstance(n, Access) for i in n.indices
                    }
                    if node.index.name not in body_idx:
                        raise ValueError(
                            "reduce index %r does not appear in its body"
                            % node.index.name
                        )

        # (c)/(d) schedule transforms must reference known indices and be sane.
        known = {idx.name for idx in self.all_indices()}
        for t in self.schedule:
            if isinstance(t, (Tile, Split, Parallelize, Unroll)):
                if t.index is not None and t.index.name not in known:
                    raise ValueError(
                        "schedule transform %r refers to unknown index %r"
                        % (t.kind, t.index.name)
                    )
            if isinstance(t, Fuse):
                for ix in (t.index_a, t.index_b):
                    if ix is not None and ix.name not in known:
                        raise ValueError(
                            "schedule transform 'fuse' refers to unknown index %r"
                            % ix.name
                        )
            if isinstance(t, (Tile, Split)):
                if getattr(t, "size", None) is not None:
                    sz = getattr(t, "size", 0) if isinstance(t, Tile) else t.factor
                    if sz <= 0:
                        raise ValueError("transform %r factor/size must be > 0" % t.kind)
            if isinstance(t, Unroll) and t.factor <= 0:
                raise ValueError("Unroll factor must be > 0 (got %d)" % t.factor)
            if isinstance(t, ThreadsPerRow) and t.k <= 0:
                raise ValueError("ThreadsPerRow must be > 0 (got %d)" % t.k)
            if isinstance(t, Vectorize) and t.width <= 0:
                raise ValueError("Vectorize width must be > 0 (got %d)" % t.width)
            if isinstance(t, Parallelize) and t.unit not in ("block", "warp", "thread"):
                raise ValueError("Parallelize unit must be block|warp|thread")

        return True

    # -- pretty printing --------------------------------------------------- #
    def pretty(self) -> str:
        """Render the IR as indented loop-nest pseudocode."""
        lines: List[str] = ["kernel %s {" % self.name]
        for st in self.stmts:
            lines.append("  assign %s = %s" % (_fmt_access(st.lhs), _fmt_expr(st.rhs)))
        lines.append("  // loop nest")
        lines.append(_pretty_nest(self))
        if self.schedule.transforms:
            lines.append("  // schedule")
            for t in self.schedule:
                lines.append("  //   " + _fmt_transform(t))
        lines.append("}")
        return "\n".join(lines)

    # -- serialization ----------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "stmts": [_stmt_to_dict(s) for s in self.stmts],
            "schedule": [_transform_to_dict(t) for t in self.schedule],
            "formats": [_format_to_dict(f) for f in self.formats],
            "dtypes": list(self.dtypes),
            "pass_log": list(self.pass_log),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Kernel":
        return Kernel(
            name=d["name"],
            stmts=[_stmt_from_dict(s) for s in d["stmts"]],
            schedule=Schedule([_transform_from_dict(t) for t in d["schedule"]]),
            formats=[_format_from_dict(f) for f in d["formats"]],
            dtypes=list(d.get("dtypes", [])),
            pass_log=list(d.get("pass_log", [])),
        )


# --------------------------------------------------------------------------- #
# Pretty helpers
# --------------------------------------------------------------------------- #
def _fmt_index(idx: Index) -> str:
    return idx.name


def _fmt_access(a: Access) -> str:
    return "%s[%s]" % (a.tensor.name, ", ".join(_fmt_index(i) for i in a.indices))


def _fmt_expr(e: Expr) -> str:
    if isinstance(e, Access):
        return _fmt_access(e)
    if isinstance(e, Const):
        return repr(e.value)
    if isinstance(e, Mul):
        return "(%s * %s)" % (_fmt_expr(e.left), _fmt_expr(e.right))
    if isinstance(e, Add):
        return "(%s + %s)" % (_fmt_expr(e.left), _fmt_expr(e.right))
    if isinstance(e, Reduce):
        return "sum(%s, %s)" % (e.index.name, _fmt_expr(e.body))
    return repr(e)  # pragma: no cover


def _pretty_nest(kernel: Kernel) -> str:
    """A readable loop-nest rendering derived from the IR structure."""
    lines: List[str] = []
    sparse = kernel.sparse_tensor()

    def _sparse_indices(t: TensorDecl) -> List[Index]:
        for st in kernel.stmts:
            for node in st.rhs.walk():
                if isinstance(node, Access) and node.tensor.name == t.name:
                    return node.indices
        return []

    def _src(idx: Index) -> str:
        if sparse is not None:
            indices = _sparse_indices(sparse)
            for pos, level in enumerate(sparse.format.levels):
                if level == "compressed" and pos < len(indices):
                    if indices[pos].name == idx.name:
                        return "%s row" % sparse.name
        return "dense"

    def _reductions(expr: Expr) -> List[Reduce]:
        """Reduction loops in the order they wrap this expression's leaves."""
        found: List[Reduce] = []
        if isinstance(expr, Reduce):
            found.append(expr)
            found.extend(_reductions(expr.body))
        elif isinstance(expr, (Mul, Add)):
            found.extend(_reductions(expr.left))
            found.extend(_reductions(expr.right))
        return found

    free = kernel.free_indices()
    depth = 1  # opening indent inside "kernel <name> {"
    if free:
        lines.append("  for %s:" % ", ".join(i.name for i in free))
        depth = 2
    for st in kernel.stmts:
        stmt_depth = depth
        for red in _reductions(st.rhs):
            lines.append(
                "%sfor %s in reduce(%s):"
                % ("  " * stmt_depth, red.index.name, _src(red.index))
            )
            stmt_depth += 1
        lines.append(
            "%sassign %s = %s"
            % ("  " * stmt_depth, _fmt_access(st.lhs), _fmt_expr(st.rhs))
        )
    if not kernel.stmts:
        lines.append("  (empty)")
    return "\n".join(lines)


def _fmt_transform(t: Transform) -> str:
    if isinstance(t, Tile):
        return "Tile(%s, %d)" % (t.index.name, t.size)
    if isinstance(t, Split):
        return "Split(%s, %d)" % (t.index.name, t.factor)
    if isinstance(t, Parallelize):
        return "Parallelize(%s, %s)" % (t.index.name, t.unit)
    if isinstance(t, Unroll):
        return "Unroll(%s, %d)" % (t.index.name, t.factor)
    if isinstance(t, Fuse):
        return "Fuse(%s, %s)" % (t.index_a.name, t.index_b.name)
    if isinstance(t, ThreadsPerRow):
        return "ThreadsPerRow(%d)" % t.k
    if isinstance(t, Vectorize):
        return "Vectorize(%d)" % t.width
    return t.kind  # pragma: no cover


# --------------------------------------------------------------------------- #
# Serialization helpers
# --------------------------------------------------------------------------- #
def _format_to_dict(f: Format) -> Dict[str, Any]:
    return {
        "name": f.name,
        "levels": list(f.levels),
        "ell_width": f.ell_width,
        "block_size": f.block_size,
    }


def _format_from_dict(d: Dict[str, Any]) -> Format:
    return Format(d["name"], tuple(d["levels"]), d.get("ell_width"), d.get("block_size"))


def _tensor_to_dict(t: TensorDecl) -> Dict[str, Any]:
    return {
        "name": t.name,
        "order": t.order,
        "format": _format_to_dict(t.format),
        "dtype": t.dtype,
        "shape": list(t.shape),
    }


def _tensor_from_dict(d: Dict[str, Any]) -> TensorDecl:
    return TensorDecl(
        name=d["name"],
        order=d["order"],
        format=_format_from_dict(d["format"]),
        dtype=d["dtype"],
        shape=tuple(d["shape"]),
    )


def _index_to_dict(i: Index) -> Dict[str, Any]:
    return {"name": i.name}


def _index_from_dict(d: Dict[str, Any]) -> Index:
    return Index(d["name"])


def _expr_to_dict(e: Expr) -> Dict[str, Any]:
    if isinstance(e, Access):
        return {"op": "access", "tensor": _tensor_to_dict(e.tensor),
                "indices": [_index_to_dict(i) for i in e.indices]}
    if isinstance(e, Const):
        return {"op": "const", "value": e.value}
    if isinstance(e, Mul):
        return {"op": "mul", "left": _expr_to_dict(e.left), "right": _expr_to_dict(e.right)}
    if isinstance(e, Add):
        return {"op": "add", "left": _expr_to_dict(e.left), "right": _expr_to_dict(e.right)}
    if isinstance(e, Reduce):
        return {"op": "reduce", "index": _index_to_dict(e.index), "body": _expr_to_dict(e.body)}
    raise TypeError("unknown expr %r" % (e,))


def _expr_from_dict(d: Dict[str, Any]) -> Expr:
    op = d["op"]
    if op == "access":
        return Access(_tensor_from_dict(d["tensor"]),
                      [_index_from_dict(i) for i in d["indices"]])
    if op == "const":
        return Const(d["value"])
    if op == "mul":
        return Mul(_expr_from_dict(d["left"]), _expr_from_dict(d["right"]))
    if op == "add":
        return Add(_expr_from_dict(d["left"]), _expr_from_dict(d["right"]))
    if op == "reduce":
        return Reduce(_index_from_dict(d["index"]), _expr_from_dict(d["body"]))
    raise TypeError("unknown expr op %r" % op)


def _stmt_to_dict(s: Stmt) -> Dict[str, Any]:
    return {"lhs": _expr_to_dict(s.lhs), "rhs": _expr_to_dict(s.rhs)}


def _stmt_from_dict(d: Dict[str, Any]) -> Stmt:
    return Stmt(lhs=_expr_from_dict(d["lhs"]), rhs=_expr_from_dict(d["rhs"]))


def _transform_to_dict(t: Transform) -> Dict[str, Any]:
    base: Dict[str, Any] = {"kind": t.kind}
    if isinstance(t, Tile):
        base.update(index=_index_to_dict(t.index), size=t.size)
    elif isinstance(t, Split):
        base.update(index=_index_to_dict(t.index), factor=t.factor)
    elif isinstance(t, Parallelize):
        base.update(index=_index_to_dict(t.index), unit=t.unit)
    elif isinstance(t, Unroll):
        base.update(index=_index_to_dict(t.index), factor=t.factor)
    elif isinstance(t, Fuse):
        base.update(index_a=_index_to_dict(t.index_a), index_b=_index_to_dict(t.index_b))
    elif isinstance(t, ThreadsPerRow):
        base.update(k=t.k)
    elif isinstance(t, Vectorize):
        base.update(width=t.width)
    return base


def _transform_from_dict(d: Dict[str, Any]) -> Transform:
    kind = d["kind"]
    if kind == "tile":
        return Tile(_index_from_dict(d["index"]), d["size"])
    if kind == "split":
        return Split(_index_from_dict(d["index"]), d["factor"])
    if kind == "parallelize":
        return Parallelize(_index_from_dict(d["index"]), d["unit"])
    if kind == "unroll":
        return Unroll(_index_from_dict(d["index"]), d["factor"])
    if kind == "fuse":
        return Fuse(_index_from_dict(d["index_a"]), _index_from_dict(d["index_b"]))
    if kind == "threads_per_row":
        return ThreadsPerRow(d["k"])
    if kind == "vectorize":
        return Vectorize(d["width"])
    raise TypeError("unknown transform kind %r" % kind)
