Search.setIndex({"docnames": ["basics/algorithms", "basics/containers", "basics/database", "basics/dictionary", "basics/executor", "basics/fields", "basics/first_kernel", "basics/freeFunctions/fill", "basics/freeFunctions/map", "basics/freeFunctions/setField", "basics/index", "basics/macros", "basics/mpi_architecture", "basics/registerclass", "basics/unstructuredMesh", "contributing", "dsl/equation", "dsl/index", "dsl/operator", "finiteVolume/cellCentred/boundaryConditions", "finiteVolume/cellCentred/index", "finiteVolume/cellCentred/operators", "finiteVolume/cellCentred/stencil", "index", "installation", "timeIntegration/forwardEuler", "timeIntegration/index", "timeIntegration/rungeKutta"], "filenames": ["basics/algorithms.rst", "basics/containers.rst", "basics/database.rst", "basics/dictionary.rst", "basics/executor.rst", "basics/fields.rst", "basics/first_kernel.rst", "basics/freeFunctions/fill.rst", "basics/freeFunctions/map.rst", "basics/freeFunctions/setField.rst", "basics/index.rst", "basics/macros.rst", "basics/mpi_architecture.rst", "basics/registerclass.rst", "basics/unstructuredMesh.rst", "contributing.rst", "dsl/equation.rst", "dsl/index.rst", "dsl/operator.rst", "finiteVolume/cellCentred/boundaryConditions.rst", "finiteVolume/cellCentred/index.rst", "finiteVolume/cellCentred/operators.rst", "finiteVolume/cellCentred/stencil.rst", "index.rst", "installation.rst", "timeIntegration/forwardEuler.rst", "timeIntegration/index.rst", "timeIntegration/rungeKutta.rst"], "titles": ["Parallel Algorithms", "&lt;no title&gt;", "Database", "Dictionary", "Executor", "Fields", "Implementing your first kernel", "<code class=\"docutils literal notranslate\"><span class=\"pre\">fill</span></code>", "<code class=\"docutils literal notranslate\"><span class=\"pre\">map</span></code>", "<code class=\"docutils literal notranslate\"><span class=\"pre\">setField</span></code>", "Basics", "Macro Definitions", "MPI Architecture", "Derived class discovery at compile time", "UnstructuredMesh", "Contributing", "Expression", "Domain Specific Language (DSL)", "Operator", "Boundary Conditions", "cellCenteredFiniteVolume", "Operators", "Stencil", "Welcome to NeoFOAM!", "Installation", "Forward Euler", "Time Integration", "Runge Kutta"], "terms": {"ar": [0, 2, 4, 5, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 23, 24, 25, 26], "basic": [0, 5, 19, 23, 26], "build": [0, 11, 18, 23, 25, 26], "block": [0, 5, 12, 24], "implement": [0, 2, 4, 5, 10, 12, 13, 14, 15, 16, 17, 18, 19, 23, 26], "advanc": [0, 25, 26], "kernel": [0, 5, 10, 19, 23], "To": [0, 2, 3, 4, 12, 13, 15, 16, 17, 18, 19, 24, 25], "simplifi": [0, 2, 5, 13, 15, 17], "we": [0, 4, 6, 12, 15, 23, 24], "provid": [0, 2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 24, 25, 26, 27], "set": [0, 2, 9, 11, 12, 15, 16, 19, 23, 24, 25, 26, 27], "standard": [0, 11, 17, 24], "These": [0, 11], "can": [0, 2, 3, 4, 6, 12, 13, 15, 17, 18, 24], "found": [0, 3, 6, 13, 15, 18], "follow": [0, 2, 3, 5, 6, 12, 13, 15, 16, 17, 19, 24, 26], "file": [0, 6, 11, 12, 15], "includ": [0, 5, 6, 12, 15], "neofoam": [0, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 16, 17, 18, 19, 24, 25, 26, 27], "core": [0, 12, 15, 16, 23], "parallelalgorithm": 0, "hpp": [0, 5, 6, 7, 8, 9, 10, 12, 15], "test": [0, 2, 3, 6, 15, 18, 23, 24], "cpp": [0, 6, 13, 15, 18], "current": [0, 5, 12, 14, 19, 26], "parallelfor": [0, 5], "parallelreduc": 0, "The": [0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 23, 24, 25, 26, 27], "code": [0, 2, 3, 5, 11, 12, 23, 24], "show": [0, 2, 5, 13], "field": [0, 2, 4, 7, 8, 9, 10, 16, 18, 19, 23, 25, 26, 27], "templat": [0, 2, 5, 7, 8, 9, 12, 13, 15, 19, 25, 27], "typenam": [0, 2, 5, 7, 8, 9, 12, 13, 19, 25, 27], "executor": [0, 5, 7, 8, 9, 10, 12, 14, 18, 19, 23], "valuetyp": [0, 7, 9, 10, 12, 15, 19], "parallelforfieldkernel": 0, "void": [0, 4, 5, 7, 8, 9, 12, 19, 25, 27], "maybe_unus": 0, "const": [0, 2, 3, 4, 5, 7, 8, 9, 12, 13, 15, 18, 19, 27], "exec": [0, 2, 4, 7, 8, 9, 18, 19], "auto": [0, 2, 4, 7, 8, 9, 12, 13, 18, 19, 25], "span": [0, 9, 12, 18], "constexpr": 0, "std": [0, 2, 3, 4, 5, 7, 8, 9, 12, 13, 19], "is_sam": 0, "remove_reference_t": 0, "serialexecutor": [0, 4, 7, 8, 9], "valu": [0, 2, 3, 5, 7, 9, 11, 12, 15, 18, 19, 24], "size_t": [0, 2, 7, 8, 9, 12, 18, 19], "i": [0, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 23, 24, 25, 26], "0": [0, 2, 7, 8, 9, 11, 12, 17, 18], "size": [0, 2, 5, 7, 8, 9, 12, 15, 25], "els": 0, "us": [0, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 18, 19, 24, 25, 27], "runon": 0, "kokko": [0, 4, 5, 12, 19, 24, 26, 27], "parallel_for": [0, 5, 19], "rangepolici": [0, 5, 19], "kokkos_lambda": [0, 5, 8, 18, 19], "base": [0, 2, 12, 13, 14, 19, 26, 27], "type": [0, 2, 3, 4, 5, 12, 15, 18, 19, 24, 25, 26, 27], "function": [0, 2, 5, 6, 7, 8, 9, 12, 13, 14, 15, 18, 19, 25], "either": [0, 4, 11, 16, 18, 24], "run": [0, 4, 15], "directli": [0, 2, 25, 26], "within": [0, 5, 12, 16, 19, 25], "loop": [0, 12], "dispatch": [0, 5], "all": [0, 2, 5, 6, 12, 13, 15, 19, 27], "non": [0, 3, 12, 15], "determin": [0, 11], "thu": [0, 5, 12, 14, 15, 16, 24], "gpu": [0, 4, 5, 12, 15, 23, 25], "gpuexecutor": [0, 4, 7, 8, 9, 19], "wa": 0, "addition": [0, 13, 24], "name": [0, 2, 5, 6, 12, 13, 15], "improv": 0, "visibl": 0, "profil": 0, "tool": [0, 13, 15, 24], "like": [0, 2, 4, 5, 15, 17, 23, 24], "nsy": 0, "final": [0, 12, 15], "assign": 0, "result": [0, 5, 12, 17], "given": [0, 7, 9, 11, 12, 15, 17], "here": [0, 2, 6, 15], "hold": [0, 5, 12, 16], "data": [0, 2, 3, 5, 12, 14, 15, 19], "pointer": [0, 11], "devic": [0, 4], "defin": [0, 2, 3, 5, 7, 8, 9, 11, 12, 13, 18], "begin": [0, 12], "end": [0, 12], "sever": [0, 5, 12, 15, 17, 24, 27], "overload": 0, "exist": [0, 2, 3, 12, 15, 23, 24], "without": [0, 3, 4, 18], "an": [0, 2, 3, 4, 5, 6, 11, 12, 13, 15, 16, 17, 18, 19], "explicitli": [0, 15, 24], "rang": [0, 7, 8, 9, 19], "learn": 0, "more": [0, 2, 3, 5, 12, 17, 18, 26], "how": [0, 2, 6, 12, 13, 16], "recommend": [0, 24], "check": [0, 3, 4, 11, 12, 13, 15], "correspond": [0, 5, 15], "unit": [0, 3, 12, 15, 23], "further": [0, 5, 12, 13, 14, 18], "detail": [0, 3, 5, 13, 14, 18, 24, 26], "free": [0, 5, 6, 25, 26], "fill": [0, 10], "map": [0, 9, 10, 12, 13], "setfield": [0, 10], "openfoam": [2, 5, 13, 16, 17, 18, 19], "object": [2, 15], "objectregistri": 2, "thei": [2, 6, 11, 12], "access": [2, 3, 12, 17], "In": [2, 5, 6, 12, 15, 16, 17, 19], "modern": [2, 23], "softwar": [2, 23], "develop": [2, 12, 17, 23, 24], "best": 2, "practic": [2, 12], "emphas": 2, "modular": [2, 17], "testabl": 2, "loos": 2, "coupl": 2, "compon": [2, 27], "while": [2, 12, 17, 25], "offer": [2, 5, 11, 14, 26], "conveni": [2, 11], "simpl": [2, 12, 17, 19, 25], "approach": [2, 4, 12, 13, 17, 23, 26], "manag": [2, 4, 12, 13, 27], "its": [2, 5, 12, 13, 17, 25], "inher": 2, "design": [2, 10, 17, 19], "conflict": 2, "principl": 2, "encourag": 2, "relianc": 2, "central": [2, 5], "contain": [2, 4, 5, 12, 14, 19, 25], "make": [2, 6, 15], "codebas": 2, "less": 2, "flexibl": [2, 26], "refactor": [2, 15], "adopt": [2, 17], "architectur": [2, 10, 13, 17, 23], "pattern": [2, 4, 6, 19], "introduc": [2, 12], "potenti": 2, "perform": [2, 5, 12, 15, 16, 17, 18], "bottleneck": 2, "debug": [2, 11, 12, 24], "challeng": 2, "contrast": [2, 17, 19], "where": [2, 4, 11, 12, 13], "each": [2, 8, 12, 25, 26], "consist": [2, 12, 15, 16], "pair": [2, 7, 8, 9], "string": [2, 3, 12, 13], "valid": 2, "ensur": [2, 12], "integr": [2, 16, 17, 23, 25, 27], "adher": [2, 15], "predefin": 2, "rule": [2, 15], "format": [2, 15, 17], "prevent": [2, 12, 24], "error": [2, 10, 12], "inconsist": 2, "tightli": 2, "fvmesh": 2, "time": [2, 4, 10, 12, 16, 17, 23, 25, 27], "class": [2, 3, 4, 6, 10, 12, 15, 16, 18, 19, 23, 25, 27], "easier": [2, 6], "diagram": 2, "illustr": 2, "relationship": 2, "between": [2, 4, 12, 16, 24, 26, 27], "A": [2, 5, 11, 12, 15, 17, 18, 19], "have": [2, 4, 12, 13, 15, 24], "n": [2, 25], "At": [2, 15], "lowest": 2, "level": [2, 23], "which": [2, 5, 12, 15, 16, 18, 19, 23], "dictionari": [2, 10, 14, 23, 25, 26, 27], "id": 2, "similar": [2, 3, 12, 13, 18], "python": [2, 3], "kei": [2, 3, 12, 13, 26, 27], "ani": [2, 3, 7, 8, 9, 12, 15], "snippet": [2, 5], "doc": [2, 13, 15], "key1": [2, 3], "value1": 2, "key2": [2, 3], "2": [2, 7, 8, 9, 12, 17, 18], "requir": [2, 5, 12, 14, 15, 17, 18, 24, 25, 26], "3": [2, 12, 16, 18], "substr": 2, "4": 2, "doc_": 2, "get": [2, 3, 15, 18], "doubl": [2, 12, 15], "mainli": [2, 5], "extend": [2, 4], "possibl": [2, 12, 16, 17, 19, 23], "dict": [2, 3], "return": [2, 5, 8, 12, 13, 15, 18], "require_nothrow": 2, "As": [2, 19], "state": [2, 15], "earlier": 2, "part": [2, 12], "itself": [2, 23], "shown": [2, 4, 5, 13, 18, 19, 24], "abov": [2, 4, 12, 17, 19], "enabl": [2, 4, 11, 12, 24], "addit": [2, 4, 5, 12, 13, 14, 16, 18], "thi": [2, 3, 4, 5, 6, 11, 12, 13, 15, 16, 17, 23, 24, 25, 26, 27], "done": [2, 12], "fielddocu": 2, "keep": 2, "refer": [2, 3, 12, 15], "metadata": 2, "index": [2, 12, 14, 15, 19, 23], "iter": [2, 5, 19], "subcycl": 2, "fieldtyp": 2, "timeindex": 2, "iterationindex": 2, "int64_t": 2, "subcycleindex": 2, "validatefielddoc": 2, "appropri": 2, "getter": [2, 15], "setter": 2, "user": [2, 12, 14, 18], "most": [2, 12, 15, 24], "regist": [2, 6, 13, 14], "fvcc": 2, "instanc": [2, 12], "db": 2, "newtestfieldcollect": 2, "volumefield": [2, 5, 18, 20, 25, 26, 27], "scalar": [2, 4, 5, 7, 8, 9, 12, 17, 18, 19, 25, 26, 27], "t": [2, 5, 8, 15, 23, 25, 27], "registerfield": 2, "createfield": 2, "mesh": [2, 5, 12, 14, 15], "1": [2, 7, 8, 9, 12, 17, 18, 25], "ha": [2, 12, 15, 17, 23, 25], "static": [2, 6, 13], "method": [2, 3, 14, 16, 17, 19, 23, 25, 26, 27], "one": [2, 12, 15, 24, 25, 27], "doesn": [2, 15], "expect": [2, 23], "createfunct": 2, "argument": [2, 4, 6, 13], "could": [2, 12, 23], "look": 2, "struct": [2, 4], "unstructuredmesh": [2, 10, 23], "oper": [2, 4, 5, 6, 12, 14, 16, 17, 19, 23, 25, 26, 27], "vector": [2, 5, 12, 16, 18, 19, 27], "volumeboundari": [2, 19, 20], "bc": [2, 20], "patchi": 2, "insert": [2, 3], "fixedvalu": 2, "push_back": 2, "internalfield": [2, 5, 19, 25], "ncell": [2, 18], "vf": 2, "easili": [2, 4, 24], "read": [2, 5, 14, 17], "allow": [2, 3, 4, 5, 12, 13, 17, 26], "u": [2, 15, 17], "find": 2, "resnam": 2, "fielddoc": 2, "constvolfield": 2, "somenam": 2, "somevalu": 2, "42": [2, 3], "lambda": [2, 5, 18], "boolean": 2, "filter": 2, "match": [2, 9], "support": [2, 5, 12, 17, 18, 26], "extens": [2, 23, 24], "through": [2, 12, 25, 26, 27], "eras": 2, "interfac": [2, 3, 4, 18, 19, 26], "minim": [2, 12, 17, 25], "onli": [2, 9, 11, 12, 17, 18, 19], "necessari": [2, 27], "add": [2, 5, 15, 18], "domain": [2, 23], "specif": [2, 3, 7, 8, 9, 10, 12, 13, 15, 18, 19, 23, 24, 27], "For": [2, 12, 15, 25, 26], "exampl": [2, 5, 6, 11, 12, 13, 15, 18], "customdocu": 2, "public": [2, 13, 19], "testvalu": 2, "validatecustomdoc": 2, "privat": [2, 15], "wrap": 2, "own": 2, "accessor": 2, "variabl": [2, 15], "below": [2, 4, 5, 12, 13, 18, 19, 24], "modifi": [2, 3, 19], "them": [2, 24], "collectionmixin": 2, "customcollect": 2, "bool": 2, "docs_": 2, "cc": 2, "fals": [2, 3, 11], "emplac": 2, "true": [2, 3], "col": 2, "inherit": [2, 15, 18, 25], "from": [2, 5, 6, 12, 13, 14, 17, 18, 23, 24, 25], "alreadi": 2, "boilerpl": 2, "focu": [2, 12, 17], "structur": [3, 5, 15], "store": [3, 5, 12, 13, 14, 15, 16, 19], "gener": [3, 11, 12, 15, 18], "deliv": [3, 23], "complex": [3, 12, 17], "input": 3, "simul": [3, 17], "need": [3, 12, 14, 16, 17, 18], "It": [3, 12, 19, 24, 25], "see": [3, 5, 6, 12], "hello": 3, "int": [3, 6, 12], "If": [3, 7, 8, 9, 12, 14, 15], "pass": [3, 4, 5, 13, 17], "well": 3, "43": 3, "throw": [3, 11, 12], "except": [3, 15], "out_of_rang": 3, "non_existent_kei": 3, "remov": [3, 12, 24], "also": [3, 5, 6, 12, 13, 15, 18], "sub": [3, 16], "group": 3, "relat": [3, 15], "togeth": [3, 12], "subdict": 3, "sdict": 3, "100": 3, "sdict2": 3, "mpi": [4, 10, 23], "x": [4, 11], "parallel": [4, 5, 10, 12, 23, 27], "execut": [4, 15, 24, 25, 27], "space": [4, 12, 17, 24, 25, 27], "memori": [4, 5, 12, 27], "specifi": [4, 11, 13, 19], "cpu": 4, "cpuexecutor": [4, 7, 8, 9], "openmp": [4, 12], "c": [4, 15, 17, 23], "thread": [4, 12], "combin": [4, 5, 12, 17, 24, 26, 27], "One": 4, "goal": [4, 23], "abil": [4, 14], "switch": [4, 24, 26], "differ": [4, 5, 12, 16, 18, 24], "e": [4, 12, 15, 16, 23], "recompil": 4, "achiev": [4, 12, 18], "gpuexec": 4, "cpuexec": 4, "gpufield": 4, "10": [4, 12, 24], "cpufield": 4, "variant": 4, "strategi": [4, 16], "alloc": [4, 5, 12], "runtim": [4, 6, 13, 26], "visit": [4, 19], "functor": 4, "cout": [4, 7, 8, 9, 12], "endl": [4, 7, 8, 9, 12], "would": [4, 12], "print": [4, 7, 8, 9, 11], "messag": [4, 11, 12], "depend": [4, 6, 24, 25, 26], "librari": [4, 11, 17, 23, 25, 26, 27], "featur": [4, 15], "should": [4, 5, 11, 12, 13, 15, 17, 24], "two": [4, 5, 11, 12, 13, 15, 26], "same": [4, 12, 17], "equal": [4, 11], "api": [5, 19, 23], "probabl": [5, 19], "chang": [5, 14, 15, 19, 24], "futur": [5, 10, 19], "capabl": [5, 26, 27], "algebra": 5, "boundaryfield": [5, 19], "friendli": 5, "datastructur": 5, "boundari": [5, 14, 20, 23], "domainfield": [5, 19], "intern": 5, "besid": 5, "finit": [5, 19], "volum": [5, 18, 19], "geometricfieldmixin": 5, "mixin": 5, "member": [5, 6, 12, 13, 14, 15, 19], "notion": 5, "concret": 5, "condiditon": 5, "surfacefield": [5, 20], "surfac": [5, 19], "equival": 5, "element": [5, 8, 9, 12], "platform": [5, 23], "portabl": [5, 23], "cfd": 5, "framework": [5, 26], "binari": [5, 23], "subtract": [5, 16], "multipl": [5, 12, 16], "some": [5, 12, 15], "nodiscard": [5, 15], "rh": 5, "exec_": 5, "size_": 5, "creat": [5, 12, 13, 15, 18, 24], "temporari": [5, 17, 18], "call": [5, 6, 12], "stand": [5, 6], "fieldfreefunct": [5, 7, 8, 9], "turn": 5, "fieldbinaryop": 5, "actual": [5, 6, 12, 19], "our": [5, 12], "ultim": [5, 16, 19], "document": [5, 10, 13, 23, 24], "type_identity_t": [5, 7, 9], "b": [5, 9, 11, 15], "va": 5, "vb": 5, "version": [5, 11], "highlight": 5, "anoth": [5, 12], "import": 5, "aspect": 5, "program": [5, 9, 11, 12, 17], "model": [5, 15], "deallocationg": 5, "finitevolum": [5, 6, 15, 19], "cellcentr": [5, 6, 15, 19], "folder": [5, 15], "namespac": 5, "both": [5, 12, 19, 24, 25, 26], "deriv": [5, 6, 10, 12, 18, 19, 23], "handl": [5, 11, 12, 16, 18, 25, 27], "geometr": [5, 12], "inform": [5, 11, 12, 14, 15, 16, 26], "via": [5, 6, 13, 23, 26], "act": [5, 19, 26], "fundament": [5, 12], "write": 5, "condit": [5, 11, 20, 23], "correctboundarycondit": [5, 19, 25], "updat": [5, 12, 14, 15, 19, 24], "": [5, 12, 13, 15, 20, 24, 26, 27], "compar": [5, 25], "volscalarfield": [5, 17], "volvectorfield": [5, 17], "voltensorfield": 5, "surfacescalarfield": 5, "surfacevectorfield": 5, "surfacetensorfield": 5, "respect": 5, "so": [5, 14, 15, 23], "branch": 5, "when": [5, 11, 12, 25, 26], "over": [5, 15, 19], "face": 5, "scalarfield": 5, "section": [6, 12, 26], "explain": 6, "gaussgreendiv": 6, "serv": [6, 23, 25, 26], "src": [6, 15], "repres": [6, 18, 19], "term": [6, 12, 16, 17, 18], "nabla": 6, "cdot": [6, 25], "phi": [6, 17], "dv": 6, "particular": 6, "dsl": [6, 16, 18, 23], "explicit": [6, 16, 18, 19, 25, 26, 27], "div": [6, 17], "henc": [6, 17], "order": [6, 12, 15, 25, 26, 27], "select": [6, 13, 14], "let": 6, "divoperatorfactori": 6, "registerclass": [6, 13], "computediv": 6, "correct": [6, 12], "common": [6, 24, 26], "sinc": [6, 12, 16], "commun": [6, 10, 23], "header": [7, 8, 9, 11, 15], "entir": [7, 9, 25], "subfield": [7, 8, 9], "option": [7, 8, 9, 24, 25, 26], "nullopt": [7, 8, 9], "paramet": [7, 8, 9, 13, 15, 19], "whole": [7, 8, 9], "other": [7, 8, 9, 12, 15, 16, 23], "copi": [7, 8, 9, 12], "host": [7, 8, 9], "hostfield": [7, 8, 9], "copytohost": [7, 8, 9], "appli": [8, 12, 15, 19], "inner": 8, "fielda": 9, "fieldb": 9, "fieldc": 9, "note": [9, 12, 24, 26], "doe": [9, 12], "exit": 9, "segfault": 9, "last": 9, "overview": 10, "cell": [10, 12, 17, 19], "centr": [10, 17], "algorithm": [10, 12, 17, 23], "boundarymesh": 10, "stencildatabas": 10, "discoveri": [10, 23], "compil": [10, 15, 23, 24], "usag": [10, 11, 26], "macro": [10, 23], "definit": [10, 18, 23], "info": 10, "background": 10, "partit": 10, "work": [10, 15, 17, 24], "databas": [10, 15, 23], "fieldcollect": 10, "queri": 10, "collect": 10, "ad": [10, 15], "new": [10, 15, 18], "your": [10, 15, 23], "first": [10, 12, 15, 19, 23, 25], "report": 11, "assert": 11, "mechan": [11, 13], "log": 11, "across": [11, 12], "nf_debug": 11, "activ": 11, "cmake_build_typ": [11, 24], "nf_debug_info": 11, "relwithdebinfo": 11, "nf_debug_messag": 11, "nf_info": 11, "output": 11, "stream": 11, "nf_dinfo": 11, "nf_error_exit": 11, "abort": 11, "nf_throw": 11, "neofoamexcept": 11, "nf_assert": 11, "nf_assert_throw": 11, "nf_debug_assert": 11, "nf_debug_assert_throw": 11, "op": [11, 12], "releas": [11, 12, 24], "nf_assert_equ": 11, "nf_debug_assert_equ": 11, "nf_assert_equal_throw": 11, "nf_debug_assert_equal_throw": 11, "nf_ping": 11, "reach": 11, "certain": 11, "line": [11, 12, 15], "pure": [11, 15], "stack": [11, 13, 15], "trace": 11, "cpptrace": 11, "critic": 11, "occur": [11, 12], "ptr": [11, 13], "nullptr": [11, 27], "null": 11, "posit": [11, 12], "virtual": [12, 18, 19], "larg": 12, "scientif": 12, "engin": [12, 17], "too": 12, "usabl": 12, "singl": 12, "comput": [12, 15, 17, 25], "problem": [12, 17], "broken": 12, "down": 12, "smaller": [12, 15, 25], "distribut": 12, "mani": 12, "case": 12, "do": [12, 14, 15, 19], "share": 12, "address": [12, 17], "solut": [12, 17, 25], "procedur": [12, 24], "rank": 12, "solv": [12, 16, 17, 25, 26, 27], "what": 12, "scalabl": [12, 18], "crucial": 12, "mask": 12, "cost": 12, "main": [12, 15, 16], "avoid": [12, 15], "up": [12, 15], "reduc": [12, 17, 18], "overhead": [12, 25], "broadli": 12, "conjunct": 12, "frequenc": 12, "discuss": [12, 15], "major": [12, 17], "brought": 12, "purpos": [12, 26], "seamlessli": 12, "suppli": 12, "typic": 12, "default": [12, 16, 24], "mpi_allreduc": 12, "reduceallscalar": 12, "reduceop": 12, "mpi_comm": 12, "comm": 12, "mpi_in_plac": 12, "reinterpret_cast": 12, "gettyp": [12, 18], "getop": 12, "reduct": 12, "automat": [12, 13, 14, 24, 25], "environ": [12, 25, 26], "locat": [12, 15], "mpiinit": 12, "mpienviron": 12, "former": 12, "raii": [12, 27], "initi": [12, 13], "destructor": 12, "start": [12, 15], "constructor": 12, "argc": 12, "char": 12, "argv": 12, "solver": [12, 17, 25], "mpi_fin": 12, "avail": [12, 24, 26], "onc": 12, "mpi_rank": 12, "mpi_siz": 12, "popul": 12, "mpi_comm_world": 12, "construct": [12, 16], "anywher": 12, "intend": 12, "split": 12, "By": [12, 23, 24], "longer": 12, "pars": 12, "With": 12, "place": 12, "mpi_commun": 12, "mpienv": 12, "sum": 12, "number": [12, 17, 18], "simplic": 12, "focus": 12, "reader": 12, "remind": 12, "terminologi": 12, "simplex": 12, "half": 12, "duplex": 12, "full": [12, 15, 24], "wai": [12, 24], "sender": 12, "receiv": 12, "vice": 12, "versa": 12, "direct": 12, "simultan": 12, "facilit": 12, "buffer": 12, "halfduplexcommbuff": 12, "respons": [12, 16, 19], "send": 12, "pun": 12, "transfer": 12, "alwai": [12, 17], "rel": 12, "expens": [12, 15], "never": 12, "laid": 12, "out": [12, 15], "continu": [12, 14, 17], "per": 12, "basi": 12, "therefor": 12, "kind": 12, "guard": 12, "rail": 12, "variou": 12, "until": 12, "finish": [12, 15], "form": [12, 17], "fullduplexcommbuff": 12, "process": [12, 15, 17], "step": [12, 15, 16, 17, 24, 25], "flag": [12, 24], "resourc": [12, 27], "load": [12, 13], "wait": 12, "unload": 12, "de": 12, "unordered_map": 12, "comm_buff": 12, "sendsiz": 12, "receives": 12, "alldata": 12, "local": [12, 15], "sendmap": 12, "assum": 12, "receivemap": 12, "obtain": 12, "initcomm": 12, "test_commun": 12, "commrank": 12, "sendbuff": 12, "getsendbuff": 12, "startcomm": 12, "waitcomplet": 12, "receivebuff": 12, "getreceivebuff": 12, "finalisecomm": 12, "later": 12, "inplac": 12, "remain": 12, "open": [12, 15, 24], "aim": [12, 15, 23], "dead": 12, "lock": 12, "detect": 12, "hang": 12, "now": 12, "shift": 12, "overlap": 12, "present": 12, "than": [12, 17], "dictat": 12, "stencil": [12, 14], "miss": 12, "neighbor": 12, "halo": 12, "enough": 12, "abl": [12, 18], "calcul": 12, "must": [12, 13, 18, 19], "reason": [12, 23], "nice": 12, "ranksimplexcommmap": 12, "arriv": 12, "Its": [12, 16], "role": 12, "pathwai": 12, "uniqu": 12, "identifi": 12, "worth": 12, "mai": [12, 25], "being": 12, "give": [12, 15], "howev": [12, 15, 17, 23, 24], "scale": [12, 16, 18], "help": [12, 16, 17], "cours": 12, "logic": [12, 19, 27], "situat": [12, 15], "made": 12, "sequenti": 12, "lead": 12, "divid": 12, "essenti": [12, 24], "formal": 12, "system": [12, 15, 16, 17, 24], "world": 12, "dynam": 12, "balanc": 12, "replac": [12, 17, 23], "metric": 12, "runtimeselectionfactori": 13, "creation": 13, "factori": 13, "explan": 13, "post": 13, "overflow": 13, "plugin": [13, 23], "unique_ptr": 13, "baseclass": 13, "derivedclass": 13, "registr": 13, "associ": 13, "take": 13, "list": [13, 15, 24], "insid": 13, "arg": 13, "registerdocument": 13, "keyexistsorerror": 13, "tabl": 13, "forward": [13, 23, 26, 27], "schema": 13, "after": [13, 15, 17, 24, 25], "been": [13, 15], "instanti": [13, 18], "testderiv": 13, "baseclassdocument": 13, "retriev": [13, 18], "baseclassnam": 13, "around": [13, 27], "relev": [14, 24], "disc": 14, "convert": 14, "foamadapt": [14, 23], "arrai": [14, 19], "patch": 14, "offset": 14, "unabl": 14, "stencildataba": 14, "placehold": 14, "yet": 14, "complet": 14, "link": 14, "highli": [15, 23], "welcom": 15, "you": [15, 24], "review": 15, "autom": 15, "enforc": 15, "clang": [15, 24], "tidi": 15, "configur": [15, 24, 25, 26, 27], "furthermor": 15, "adequ": 15, "licens": 15, "sourc": [15, 23, 25], "reus": 15, "linter": 15, "typo": 15, "obviou": 15, "spell": 15, "issu": [15, 17], "stylist": 15, "rather": [15, 16, 17], "advic": 15, "ambigu": 15, "mention": 15, "ration": 15, "decis": 15, "try": 15, "compli": 15, "guidelin": 15, "camelcas": 15, "capit": 15, "descript": [15, 24], "prefer": 15, "float": 15, "indic": 15, "simpli": 15, "instead": [15, 19, 23], "omit": 15, "abstract": 15, "flat": 15, "hierarchi": 15, "composit": 15, "unintend": 15, "advis": 15, "might": [15, 23], "unstructur": 15, "outliv": 15, "suffix": 15, "_": 15, "g": [15, 16], "foo": 15, "inout": 15, "variat": 15, "redund": 15, "ie": 15, "geometrymodel": 15, "finitevolumecellcentredgeometrymodel": 15, "want": [15, 23], "fix": [15, 19], "pleas": 15, "don": [15, 23], "hesit": 15, "pr": 15, "person": 15, "readi": [15, 23], "least": 15, "ideal": 15, "approv": 15, "befor": 15, "merg": 15, "sure": 15, "pipelin": 15, "succe": 15, "suffici": 15, "point": 15, "ci": 15, "hardwar": 15, "bug": 15, "entri": 15, "changelog": 15, "md": 15, "skip": 15, "permiss": 15, "rebas": 15, "latest": [15, 24], "yourself": 15, "author": 15, "small": 15, "medium": 15, "exce": 15, "1000": 15, "consid": 15, "break": 15, "influenc": 15, "mean": [15, 23, 25], "signal": 15, "whether": 15, "command": [15, 24], "cach": 15, "forc": 15, "rebuild": 15, "everi": 15, "push": 15, "aw": 15, "doxygen": [15, 24], "onlin": 15, "sphinx": [15, 24], "instal": [15, 23], "second": 15, "cmake": [15, 23], "dneofoam_build_doc": 15, "ON": [15, 24], "target": 15, "html": 15, "docs_build": [15, 24], "built": 15, "directori": [15, 24], "view": 15, "web": 15, "browser": 15, "firefox": 15, "altern": 15, "just": 15, "formul": 16, "equat": [16, 17, 18, 25, 26, 27], "li": 16, "answer": 16, "question": 16, "discret": 16, "spatial": 16, "fvscheme": [16, 17], "fvsolut": [16, 17], "evalu": [16, 17], "lazili": 16, "ti": 16, "delai": 16, "numer": [16, 17], "rk": [16, 17, 27], "even": 16, "lazi": [16, 17], "implicit": [16, 18, 19], "tempor": [16, 18], "consequ": [16, 18], "concept": 17, "express": [17, 18, 23, 25, 27], "concis": 17, "readabl": 17, "close": 17, "resembl": 17, "mathemat": 17, "represent": [17, 27], "littl": 17, "knowledg": 17, "scheme": 17, "physic": 17, "effort": 17, "maintain": 17, "navier": 17, "stoke": 17, "fvvectormatrix": 17, "ueqn": 17, "fvm": 17, "ddt": 17, "laplacian": 17, "nu": 17, "fvc": 17, "grad": 17, "p": [17, 24], "piso": 17, "vectormatrix": 17, "diagon": 17, "off": [17, 24], "matrix": 17, "rau": 17, "hbya": 17, "constrainhbya": 17, "h": 17, "easi": 17, "understand": 17, "familiar": 17, "limit": 17, "due": 17, "spars": 17, "individu": 17, "eagerli": 17, "unnecessari": 17, "ldu": 17, "extern": [17, 25, 26], "linear": 17, "discretis": 17, "tri": 17, "better": 17, "optimis": 17, "coo": 17, "csr": 17, "pde": [17, 26], "sundial": [17, 26, 27], "bdf": 17, "heterogen": 17, "drop": 17, "imp": 17, "exp": [17, 27], "assembli": 17, "defer": 17, "till": 17, "dure": 17, "That": 17, "assembl": 17, "coeffici": 18, "erasur": 18, "polymorph": 18, "divterm": 18, "diverg": 18, "ddtterm": 18, "timeterm": 18, "fit": 18, "storag": 18, "scalingfield": 18, "sf": 18, "customterm": 18, "constantscaledterm": 18, "constant": 18, "factor": 18, "fieldscaledterm": 18, "syntax": 18, "multiscaledterm": 18, "lambdascaledterm": 18, "operatormixin": 18, "explicitoper": [18, 25], "implicitoper": 18, "coeff": 18, "getcoeffici": 18, "draft": 19, "underli": 19, "noop": 19, "surfaceboundari": [19, 20], "attribut": 19, "boundarypatchmixin": 19, "center": 19, "volumetr": 19, "visitor": 19, "fvccscalarfixedvalueboundaryfield": 19, "bfield": 19, "fixedvaluebckernel": 19, "kernel_": 19, "mesh_": 19, "patchid_": 19, "start_": 19, "end_": 19, "uniformvalue_": 19, "s_valu": 19, "s_refvalu": 19, "refvalu": 19, "uniformvalu": 19, "contigu": 19, "uniform": 19, "volfield": 19, "project": [23, 24], "bring": 23, "reimplement": 23, "libfinitevolum": 23, "libopenfoam": 23, "compliant": 23, "20": 23, "high": [23, 27], "interoper": 23, "deviat": 23, "driven": 23, "contribut": 23, "everyon": 23, "preset": 23, "prerequisit": 23, "workflow": 23, "vscode": 23, "style": 23, "guid": 23, "collabor": 23, "pull": 23, "request": 23, "github": [23, 24], "label": 23, "cellcenteredfinitevolum": 23, "languag": 23, "euler": [23, 26, 27], "rung": [23, 26], "kutta": [23, 26], "abi": 23, "won": 23, "produc": 23, "applic": [23, 26], "pimplefoam": 23, "against": 23, "demonstr": 23, "repositori": [23, 24], "modul": [23, 27], "search": 23, "page": 23, "clone": 24, "git": 24, "http": 24, "com": 24, "exasim": 24, "navig": 24, "cd": 24, "mkdir": 24, "desiredbuildflag": 24, "chain": 24, "d": 24, "mode": 24, "neofoam_build_doc": 24, "neofoam_build_test": 24, "brows": 24, "ccmake": 24, "gui": 24, "prefix": 24, "neofoam_": 24, "kokkos_enable_cuda": 24, "kokkos_enable_hip": 24, "commonli": 24, "product": 24, "ninja": 24, "chosen": 24, "bash": 24, "sudo": 24, "apt": 24, "pip": 24, "pre": 24, "commit": 24, "furo": 24, "breath": 24, "sitemap": 24, "ubuntu": 24, "24": 24, "04": 24, "16": 24, "gcc": 24, "libomp": 24, "dev": 24, "python3": 24, "14": 24, "rm": 24, "usr": 24, "bin": 24, "ln": 24, "m": 24, "cpptool": 24, "button": 24, "tab": 24, "flask": 24, "icon": 24, "task": 24, "menu": 24, "ctrl": 24, "press": 24, "enter": 24, "y_": 25, "y_n": 25, "delta": 25, "f": 25, "t_n": 25, "self": 25, "forwardeul": [25, 26], "timeintegratorbas": [25, 26], "straightforward": 25, "solutionfieldtyp": [25, 27], "eqn": 25, "sol": 25, "dt": [25, 27], "lightweight": [25, 26], "guarante": 25, "regardless": 25, "constraint": 25, "timedict": [25, 26, 27], "timeintegr": [25, 26, 27], "timestep": [25, 27], "solutionfield": [25, 27], "currenttim": [25, 27], "deltat": [25, 27], "fallback": [25, 26], "unavail": 25, "accuraci": 25, "higher": [25, 26], "synchron": 25, "partial": 26, "differenti": 26, "distinct": 26, "nativ": 26, "demand": 26, "robust": [26, 27], "seamless": 26, "wip": 26, "about": 26, "consider": 26, "leverag": 27, "rungekutta": 27, "wrapper": 27, "erkstep": 27, "convers": 27, "pdeexpr_": 27, "initsunerksolv": 27, "context": 27, "choos": 27}, "objects": {"": [[7, 0, 1, "_CPPv4I0EN7NeoFOAM4fillEvR5FieldI9ValueTypeEKNSt15type_identity_tI9ValueTypeEENSt8optionalINSt4pairI6size_t6size_tEEEE", "NeoFOAM::fill"], [7, 1, 1, "_CPPv4I0EN7NeoFOAM4fillEvR5FieldI9ValueTypeEKNSt15type_identity_tI9ValueTypeEENSt8optionalINSt4pairI6size_t6size_tEEEE", "NeoFOAM::fill::ValueType"], [7, 2, 1, "_CPPv4I0EN7NeoFOAM4fillEvR5FieldI9ValueTypeEKNSt15type_identity_tI9ValueTypeEENSt8optionalINSt4pairI6size_t6size_tEEEE", "NeoFOAM::fill::a"], [7, 2, 1, "_CPPv4I0EN7NeoFOAM4fillEvR5FieldI9ValueTypeEKNSt15type_identity_tI9ValueTypeEENSt8optionalINSt4pairI6size_t6size_tEEEE", "NeoFOAM::fill::range"], [7, 2, 1, "_CPPv4I0EN7NeoFOAM4fillEvR5FieldI9ValueTypeEKNSt15type_identity_tI9ValueTypeEENSt8optionalINSt4pairI6size_t6size_tEEEE", "NeoFOAM::fill::value"], [19, 3, 1, "_CPPv4I0EN7NeoFOAM12finiteVolume11cellCentred15SurfaceBoundaryE", "NeoFOAM::finiteVolume::cellCentred::SurfaceBoundary"], [19, 1, 1, "_CPPv4I0EN7NeoFOAM12finiteVolume11cellCentred15SurfaceBoundaryE", "NeoFOAM::finiteVolume::cellCentred::SurfaceBoundary::ValueType"], [19, 3, 1, "_CPPv4I0EN7NeoFOAM12finiteVolume11cellCentred14VolumeBoundaryE", "NeoFOAM::finiteVolume::cellCentred::VolumeBoundary"], [19, 1, 1, "_CPPv4I0EN7NeoFOAM12finiteVolume11cellCentred14VolumeBoundaryE", "NeoFOAM::finiteVolume::cellCentred::VolumeBoundary::ValueType"], [8, 0, 1, "_CPPv4I00EN7NeoFOAM3mapEvR5FieldI1TEK5InnerNSt8optionalINSt4pairI6size_t6size_tEEEE", "NeoFOAM::map"], [8, 1, 1, "_CPPv4I00EN7NeoFOAM3mapEvR5FieldI1TEK5InnerNSt8optionalINSt4pairI6size_t6size_tEEEE", "NeoFOAM::map::Inner"], [8, 1, 1, "_CPPv4I00EN7NeoFOAM3mapEvR5FieldI1TEK5InnerNSt8optionalINSt4pairI6size_t6size_tEEEE", "NeoFOAM::map::T"], [8, 2, 1, "_CPPv4I00EN7NeoFOAM3mapEvR5FieldI1TEK5InnerNSt8optionalINSt4pairI6size_t6size_tEEEE", "NeoFOAM::map::a"], [8, 2, 1, "_CPPv4I00EN7NeoFOAM3mapEvR5FieldI1TEK5InnerNSt8optionalINSt4pairI6size_t6size_tEEEE", "NeoFOAM::map::inner"], [8, 2, 1, "_CPPv4I00EN7NeoFOAM3mapEvR5FieldI1TEK5InnerNSt8optionalINSt4pairI6size_t6size_tEEEE", "NeoFOAM::map::range"], [9, 0, 1, "_CPPv4I0EN7NeoFOAM8setFieldEvR5FieldI9ValueTypeEKNSt4spanIKNSt15type_identity_tI9ValueTypeEEEENSt8optionalINSt4pairI6size_t6size_tEEEE", "NeoFOAM::setField"], [9, 1, 1, "_CPPv4I0EN7NeoFOAM8setFieldEvR5FieldI9ValueTypeEKNSt4spanIKNSt15type_identity_tI9ValueTypeEEEENSt8optionalINSt4pairI6size_t6size_tEEEE", "NeoFOAM::setField::ValueType"], [9, 2, 1, "_CPPv4I0EN7NeoFOAM8setFieldEvR5FieldI9ValueTypeEKNSt4spanIKNSt15type_identity_tI9ValueTypeEEEENSt8optionalINSt4pairI6size_t6size_tEEEE", "NeoFOAM::setField::a"], [9, 2, 1, "_CPPv4I0EN7NeoFOAM8setFieldEvR5FieldI9ValueTypeEKNSt4spanIKNSt15type_identity_tI9ValueTypeEEEENSt8optionalINSt4pairI6size_t6size_tEEEE", "NeoFOAM::setField::b"], [9, 2, 1, "_CPPv4I0EN7NeoFOAM8setFieldEvR5FieldI9ValueTypeEKNSt4spanIKNSt15type_identity_tI9ValueTypeEEEENSt8optionalINSt4pairI6size_t6size_tEEEE", "NeoFOAM::setField::range"]]}, "objtypes": {"0": "cpp:function", "1": "cpp:templateParam", "2": "cpp:functionParam", "3": "cpp:class"}, "objnames": {"0": ["cpp", "function", "C++ function"], "1": ["cpp", "templateParam", "C++ template parameter"], "2": ["cpp", "functionParam", "C++ function parameter"], "3": ["cpp", "class", "C++ class"]}, "titleterms": {"parallel": 0, "algorithm": 0, "databas": 2, "fieldcollect": 2, "queri": 2, "document": [2, 15], "collect": 2, "ad": 2, "new": 2, "creat": 2, "custom": 2, "store": 2, "dictionari": 3, "executor": 4, "overview": [4, 5, 11], "design": 4, "field": [5, 12], "The": 5, "valuetyp": 5, "class": [5, 13], "cell": 5, "centr": 5, "specif": [5, 17], "implement": [6, 25, 27], "your": 6, "first": 6, "kernel": 6, "fill": 7, "descript": [7, 8, 9], "definit": [7, 8, 9, 11], "exampl": [7, 8, 9], "map": 8, "setfield": 9, "basic": 10, "macro": 11, "info": 11, "hpp": 11, "error": 11, "mpi": 12, "architectur": 12, "background": 12, "commun": 12, "wrap": 12, "global": 12, "point": 12, "synchron": 12, "partit": 12, "futur": 12, "work": 12, "deriv": 13, "discoveri": 13, "compil": 13, "time": [13, 26], "usag": [13, 25, 27], "unstructuredmesh": 14, "boundarymesh": 14, "stencildatabas": 14, "contribut": 15, "neofoam": [15, 23], "code": 15, "style": 15, "guid": 15, "collabor": 15, "via": 15, "pull": 15, "request": 15, "github": 15, "workflow": [15, 24], "label": 15, "build": [15, 24], "express": 16, "domain": 17, "languag": 17, "dsl": 17, "oper": [18, 21], "boundari": 19, "condit": 19, "volumefield": 19, "": 19, "bc": 19, "surfacefield": 19, "cellcenteredfinitevolum": 20, "stencil": 22, "welcom": 23, "tabl": 23, "content": 23, "compat": 23, "openfoam": 23, "indic": 23, "instal": 24, "cmake": 24, "preset": 24, "prerequisit": 24, "vscode": 24, "forward": 25, "euler": 25, "consider": 25, "integr": 26, "rung": 27, "kutta": 27}, "envversion": {"sphinx.domains.c": 3, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 9, "sphinx.domains.index": 1, "sphinx.domains.javascript": 3, "sphinx.domains.math": 2, "sphinx.domains.python": 4, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinx.ext.todo": 2, "sphinx.ext.viewcode": 1, "sphinx": 60}, "alltitles": {"Parallel Algorithms": [[0, "parallel-algorithms"]], "Database": [[2, "database"]], "FieldCollection": [[2, "fieldcollection"]], "Query of document in a collection": [[2, "query-of-document-in-a-collection"]], "Adding a new collection and documents to the database": [[2, "adding-a-new-collection-and-documents-to-the-database"]], "Creating a Custom Document": [[2, "creating-a-custom-document"]], "Storing Custom Documents in a Custom Collection": [[2, "storing-custom-documents-in-a-custom-collection"]], "Dictionary": [[3, "dictionary"]], "Executor": [[4, "executor"]], "Overview": [[4, "overview"], [5, "overview"], [11, "overview"]], "Design": [[4, "design"]], "Fields": [[5, "fields"]], "The Field<ValueType> class": [[5, "the-field-valuetype-class"]], "Cell Centred Specific Fields": [[5, "cell-centred-specific-fields"]], "Implementing your first kernel": [[6, "implementing-your-first-kernel"]], "fill": [[7, "fill"]], "Description": [[7, "description"], [8, "description"], [9, "description"]], "Definition": [[7, "definition"], [8, "definition"], [9, "definition"]], "Example": [[7, "example"], [8, "example"], [9, "example"]], "map": [[8, "map"]], "setField": [[9, "setfield"]], "Basics": [[10, "basics"]], "Macro Definitions": [[11, "macro-definitions"]], "Info.hpp": [[11, "info-hpp"]], "Error.hpp": [[11, "error-hpp"]], "MPI Architecture": [[12, "mpi-architecture"]], "Background": [[12, "background"]], "Communication": [[12, "communication"]], "MPI Wrapping": [[12, "mpi-wrapping"]], "Global Communication": [[12, "global-communication"]], "Point-to-Point Communication": [[12, "point-to-point-communication"]], "Field Synchronization": [[12, "field-synchronization"]], "Partitioning": [[12, "partitioning"]], "Future Work": [[12, "future-work"]], "Derived class discovery at compile time": [[13, "derived-class-discovery-at-compile-time"]], "Usage": [[13, "usage"], [25, "usage"], [27, "usage"]], "UnstructuredMesh": [[14, "unstructuredmesh"]], "BoundaryMesh": [[14, "boundarymesh"]], "StencilDataBase": [[14, "stencildatabase"]], "Contributing": [[15, "contributing"]], "NeoFOAM Code Style Guide": [[15, "neofoam-code-style-guide"]], "Collaboration via Pull Requests": [[15, "collaboration-via-pull-requests"]], "Github Workflows and Labels": [[15, "github-workflows-and-labels"]], "Building the Documentation": [[15, "building-the-documentation"]], "Expression": [[16, "expression"]], "Domain Specific Language (DSL)": [[17, "domain-specific-language-dsl"]], "Operator": [[18, "operator"]], "Boundary Conditions": [[19, "boundary-conditions"]], "Boundary Conditions for VolumeField\u2019s": [[19, "boundary-conditions-for-volumefield-s"]], "BC for surfaceField": [[19, "bc-for-surfacefield"]], "cellCenteredFiniteVolume": [[20, "cellcenteredfinitevolume"]], "Operators": [[21, "operators"]], "Stencil": [[22, "stencil"]], "Welcome to NeoFOAM!": [[23, "welcome-to-neofoam"]], "Table of Contents": [[23, "table-of-contents"]], "Compatibility with OpenFOAM": [[23, "compatibility-with-openfoam"]], "Indices and tables": [[23, "indices-and-tables"]], "Installation": [[24, "installation"]], "Building with CMake Presets": [[24, "building-with-cmake-presets"]], "Prerequisites": [[24, "prerequisites"]], "Workflow with vscode": [[24, "workflow-with-vscode"]], "Forward Euler": [[25, "forward-euler"]], "Implementation": [[25, "implementation"], [27, "implementation"]], "Considerations": [[25, "considerations"]], "Time Integration": [[26, "time-integration"]], "Runge Kutta": [[27, "runge-kutta"]]}, "indexentries": {"neofoam::fill (c++ function)": [[7, "_CPPv4I0EN7NeoFOAM4fillEvR5FieldI9ValueTypeEKNSt15type_identity_tI9ValueTypeEENSt8optionalINSt4pairI6size_t6size_tEEEE"]], "neofoam::map (c++ function)": [[8, "_CPPv4I00EN7NeoFOAM3mapEvR5FieldI1TEK5InnerNSt8optionalINSt4pairI6size_t6size_tEEEE"]], "neofoam::setfield (c++ function)": [[9, "_CPPv4I0EN7NeoFOAM8setFieldEvR5FieldI9ValueTypeEKNSt4spanIKNSt15type_identity_tI9ValueTypeEEEENSt8optionalINSt4pairI6size_t6size_tEEEE"]], "neofoam::finitevolume::cellcentred::surfaceboundary (c++ class)": [[19, "_CPPv4I0EN7NeoFOAM12finiteVolume11cellCentred15SurfaceBoundaryE"]], "neofoam::finitevolume::cellcentred::volumeboundary (c++ class)": [[19, "_CPPv4I0EN7NeoFOAM12finiteVolume11cellCentred14VolumeBoundaryE"]]}})