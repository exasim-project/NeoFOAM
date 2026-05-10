# NeoN Conventions

**Analysis Date:** 2026-05-10

---

## File Headers

Every `.hpp` and `.cpp` file must begin with REUSE-compliant SPDX headers. This is enforced by the `reuse` pre-commit hook:

```cpp
// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT
```

The pre-commit hook (`reuse-annotate-mit`) auto-adds these headers. Missing or incorrect headers cause CI failure.

---

## Naming

### Files

- Header files use `lowerCamelCase.hpp` (e.g., `volumeField.hpp`, `parallelAlgorithms.hpp`, `runtimeSelectionFactory.hpp`)
- Source files use `lowerCamelCase.cpp` (e.g., `gaussGreenDiv.cpp`, `faceToMatrixAddress.cpp`)
- Test files mirror the module name: `test/<module>/<feature>.cpp` (e.g., `test/finiteVolume/cellCentred/operator/gaussGreenDiv.cpp`)
- Headers live in `include/NeoN/<module>/` and mirror the `src/<module>/` directory layout

### Classes and Structs

- `CamelCase` (PascalCase) for all class, struct, enum, and union names
- Examples: `VolumeField`, `SpatialOperator`, `SerialExecutor`, `DimensionMismatch`, `ExecutorGenerator`, `ApproxScalar`

### Functions and Methods

- `camelBack` (lowerCamelCase) for all functions and methods
- Examples: `parallelFor`, `createDefaultExecutor`, `correctBoundaryConditions`, `copyToHost`, `explicitOperation`
- Free standing functions follow the same rule: `fill`, `fence`, `executorName`

### Variables and Parameters

- `camelBack` for local variables, function parameters, and member variables
- Private member variables use a trailing underscore: `size_`, `data_`, `exec_`, `model_`, `what_`
- Examples from `Vector`: `size_`, `data_`, `exec_`; from `SpatialOperator`: `model_`

### Template Parameters

- Single-letter uppercase for type parameters used inline: `T`, `ValueType` when named
- Concept-constrained template parameters use descriptive `CamelCase` names: `Kernel`, `ExecutorType`, `IsSpatialOperator`

### Concepts

- `CamelCase` verb phrases or predicates: `parallelForKernel`, `HasExplicitOperator`, `HasImplicitOperator`, `IsSpatialOperator`
- Defined in the same header as the primary template they constrain

### Macros

- `UPPER_CASE` with `NEON_` prefix for portability macros: `NEON_LAMBDA`, `NEON_INLINE_FUNCTION`
- `NN_` prefix for type-iteration macros: `NN_FOR_ALL_VALUE_TYPES`, `NN_FOR_ALL_SCALAR_TYPES`, `NN_FOR_ALL_INTEGER_TYPES`
- Assertion macros use `NeoN_` prefix: `NeoN_ASSERT_EQUAL_LENGTH`

### Namespaces

- Top-level: `NeoN`
- Nested namespaces use C++17 syntax: `namespace NeoN::finiteVolume::cellCentred { ... }`
- Short aliases are defined locally in source files and tests: `namespace fvcc = NeoN::finiteVolume::cellCentred;`, `namespace dsl = NeoN::dsl;`
- The `la` alias is defined in headers: `namespace la = NeoN::la;`

---

## Executor Threading

**Rule: Never hard-code a concrete executor type. Always accept `const Executor& exec` as a parameter and thread it through the call stack.**

The `Executor` type (`include/NeoN/core/executor/executor.hpp`) is:
```cpp
using Executor = std::variant<SerialExecutor, CPUExecutor, GPUExecutor>;
```

All data containers (`Array`, `Vector`, `VolumeField`, `SurfaceField`) carry the executor and use it for allocation and kernel dispatch.

### Correct Pattern — executor passed through

```cpp
// Constructor accepting executor
VolumeField<scalar> phi(exec, "phi", mesh, bcs);

// Free function accepting executor
NeoN::Vector<scalar> result(exec, phi.size());
NeoN::fill(result, zero<scalar>());
```

### Incorrect Pattern — hard-coded executor

```cpp
// WRONG: hard-codes CPU serial execution
NeoN::Vector<scalar> result(NeoN::SerialExecutor{}, phi.size());
```

### Dispatch via std::visit

Executor-polymorphic code uses `std::visit` or `std::holds_alternative`:

```cpp
// Using std::holds_alternative for serial fast-path
if (std::holds_alternative<SerialExecutor>(exec))
{
    for (localIdx i = 0; i < n; i++) { /* serial loop */ }
}
else
{
    using runOn = typename ExecutorType::exec;
    Kokkos::parallel_for(name, Kokkos::RangePolicy<runOn>(start, end), kernel);
}

// Using std::visit for generic dispatch
inline std::string executorName(const Executor& exec)
{
    return std::visit(
        []<typename Exec>(const Exec& concExec) { return concExec.name(); }, exec
    );
}
```

### Creating the "best available" executor

Use `createDefaultExecutor()` when no caller-provided executor is available — it selects GPU > CPU > Serial based on compiled Kokkos backends:

```cpp
auto exec = NeoN::createDefaultExecutor();
```

In tests, `allAvailableExecutor()` (from `test/catch2/executorGenerator.hpp`) generates all compiled backends:

```cpp
auto [execName, exec] = GENERATE(allAvailableExecutor());
```

---

## C++20 Features Used

### Concepts

Concepts enforce interface requirements on template parameters without virtual dispatch:

```cpp
// From include/NeoN/core/parallelAlgorithms.hpp
template<typename Kernel>
concept parallelForKernel = requires(Kernel t, size_t i) {
    { t(i) } -> std::same_as<void>;
};

// From include/NeoN/dsl/spatialOperator.hpp
template<typename T>
concept HasExplicitOperator = requires(T const t) {
    { t.explicitOperation(std::declval<Vector<typename T::VectorValueType>&>()) }
        -> std::same_as<void>;
};

template<typename T>
concept IsSpatialOperator = HasExplicitOperator<T> || HasImplicitOperator<T>;
```

Concepts are the primary interface contract for DSL operators and kernel callables.

### Structured Bindings

Used throughout tests and production code:

```cpp
auto [execName, exec] = GENERATE(allAvailableExecutor());
auto [start, end] = range;
auto [numIter, initResNorm, finalResNorm, solveTime] = solverStats.entries[0];
```

### C++17 Nested Namespace Declarations (adopted)

```cpp
namespace NeoN::finiteVolume::cellCentred { ... }
```

### if constexpr

Used for compile-time executor dispatch:

```cpp
if constexpr (std::is_same<std::remove_reference_t<ExecutorType>, SerialExecutor>::value)
{
    /* serial path */
}
```

### Template Lambda (C++20)

```cpp
std::visit(
    []<typename Exec>(const Exec& concExec) { return concExec.name(); }, exec
);
```

---

## Key Code Patterns

### Parallel Kernels — NEON_LAMBDA

The `NEON_LAMBDA` macro expands to `KOKKOS_LAMBDA` when Kokkos is available, adding `__host__ __device__` annotations required for GPU kernels:

```cpp
parallelFor(
    exec,
    {0, nCells},
    NEON_LAMBDA(const localIdx i) { result[i] = value; }
);
```

Never use `[&]` directly in kernels intended to run on GPU — always use `NEON_LAMBDA`.

### Field Creation Pattern

```cpp
// 1. Create boundary conditions
auto bcs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(mesh);

// 2. Construct field on executor
fvcc::VolumeField<scalar> phi(exec, "phi", mesh, bcs);

// 3. Fill internal and boundary data
NeoN::fill(phi.internalVector(), 1.0);
NeoN::fill(phi.boundaryData().value(), 1.0);

// 4. Sync boundary conditions
phi.correctBoundaryConditions();
```

### Operator Application (DSL)

```cpp
// Build expression
auto expr = dsl::imp::div(phi, U) - dsl::imp::laplacian(gamma, U);

// Supply scheme configuration
expr.read(inputDict);

// Assemble linear system
auto [sp, ls] = expr.assemble(mesh, dt, relaxFactor);
```

Explicit vs implicit operators are selected by namespace: `dsl::exp::` for explicit, `dsl::imp::` for implicit.

### Type Erasure for Operators

`SpatialOperator<T>` (`include/NeoN/dsl/spatialOperator.hpp`) wraps any type satisfying `IsSpatialOperator` via internal `OperatorModel<T>`. The wrapped type is heap-allocated and accessed through a virtual interface. This allows heterogeneous operators in a `std::vector<SpatialOperator<T>>`.

### RuntimeSelectionFactory

Schemes and boundary conditions are registered using a static bool trick and selected by string name from `Dictionary`:

```cpp
// Registration (in derived class .cpp)
static bool REGISTERED = BaseClass::registerClass("schemeName", [](...){ return std::make_unique<Derived>(...); });

// Selection
auto scheme = BaseClass::create("schemeName", ...);
```

### Dictionary / Input Duality

`Input` is `std::variant<Dictionary, TokenList>`. Operators accept `Input` to allow both structured dicts and flat token lists:

```cpp
// Token list form (matches OF-style "Gauss linear")
Input input = TokenList({std::string("Gauss"), std::string("linear")});

// Dictionary form
Input input = Dictionary({
    {"DivOperator", std::string("Gauss")},
    {"surfaceInterpolation", std::string("linear")}
});
```

### Device-to-Host Copy for Inspection

`Vector` data is device-resident. To inspect values in tests or host code, call `copyToHost()`:

```cpp
auto hostData = field.copyToHost();
for (auto value : hostData.view()) { /* safe to read */ }
```

Never take a `view()` of a device `Vector` and dereference it on the host.

---

## Formatting Rules (.clang-format)

Enforced by clang-format v17 via pre-commit hook.

| Rule | Value |
|------|-------|
| Column limit | 100 |
| Indent width | 4 spaces (no tabs) |
| Pointer/reference alignment | Left (`int* p`, `int& r`) |
| Brace style | Allman (custom): braces always on their own line |
| Template declarations | Always break before `template<...>` |
| Empty line after access modifier | Always (`public:` followed by blank line) |
| `AlignAfterOpenBracket` | BlockIndent |
| `BreakBeforeBinaryOperators` | NonAssignment |
| `SortIncludes` | false — do not reorder includes |
| Short functions on single line | Allowed |
| Short lambdas on single line | Allowed |

---

## Documentation

- Doxygen-style `/** @brief ... */` comment blocks on all public classes, methods, and non-trivial free functions
- `@param`, `@tparam`, `@return`, `@throw` tags used consistently
- Inline members annotated with `//!< description` (Doxygen member doc comment)
- Math typeset with LaTeX `\f[ ... \f]` notation in Doxygen comments

---

## Error Handling

- Custom exception hierarchy in `include/NeoN/helpers/exceptions.hpp`
- Base class `NeoN::Error` stores `file:line: message`
- `NeoN::DimensionMismatch` for size mismatches
- Assertion macro `NeoN_ASSERT_EQUAL_LENGTH(a, b)` throws `DimensionMismatch`
- Tests use `REQUIRE_THROWS_AS(expr, std::runtime_error)` to verify error paths

---

## What to Avoid

**Hard-coding `SerialExecutor`** — breaks GPU portability. Use `createDefaultExecutor()` or accept `const Executor& exec` from the caller.

**Using `[&]` captures in parallel kernels** — GPU kernels require `NEON_LAMBDA` (expands to `KOKKOS_LAMBDA`) for `__device__` annotation.

**Dereferencing device `Vector::view()` on the host** — always call `copyToHost()` before accessing data outside a kernel.

**Using `std::sort`, `std::transform`, etc. on device data** — these are host STL algorithms and undefined behavior on device pointers. Use `parallelFor` or Kokkos algorithms instead.

**Reordering includes** — `SortIncludes: false` in clang-format is intentional; OpenFOAM includes are order-sensitive.

**Abbreviating the `NeoN::` namespace in headers** — `using namespace NeoN;` in headers pollutes the global namespace of downstream consumers.

**Registering schemes with names that differ from OpenFOAM canonical spelling** — scheme names registered in `RuntimeSelectionFactory` must match exactly what appears in `fvSchemes` dictionaries.

**Leaving temporary `#define CATCH_CONFIG_RUNNER` in shared headers** — this define belongs only in the one `.cpp` that provides the test `main`. The shared `catch2_common.hpp` does not include it.

---

*Convention analysis: 2026-05-10*
