# GSoC 2026 - ROOT/TMVA SOFIE Evaluation Exercises
**Submission Date**: March 14, 2026  
**System**: macOS 26.3 (Sequoia), Apple M1, arm64 architecture  
**ROOT Version**: 6.38.02

---

## Exercise 1: Building ROOT from Source

### Build Configuration

**Configuration Command:**
```bash
cmake -G Ninja -S . -B build_dir \
  -DCMAKE_INSTALL_PREFIX="$(pwd)/install_dir/" \
  -Dtmva-sofie=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64
```

**Configuration Rationale:**
- **Ninja Generator**: Faster parallel builds compared to Make, better suited for large projects like ROOT
- **TMVA SOFIE Enabled**: Required for exercises - enables ONNX model inference code generation
- **arm64 Architecture**: Native Apple Silicon compilation for optimal performance


### Build Process

**Build Command:**
```bash
cmake --build build_dir -j8
```

**Build Statistics:**
- Total compilation units: 5,840
- Build time: ~50 minutes (8 parallel jobs)
- Build system: Ninja with parallel compilation
- No fatal errors encountered
- Some warnings expected (large legacy codebase)

**Installation Command:**
```bash
cmake --install build_dir
```

**Installation Location:** `$(pwd)/install_dir/`

### Verification

**ROOT Environment Setup:**
```bash
source install_dir/bin/thisroot.sh
```

**Version Verification:**
```bash
root --version
# Output: ROOT Version: 6.38.02
```

**Feature Verification:**
```bash
root-config --features | grep sofie
# Confirms: tmva-sofie enabled
```

**Protobuf Integration Check:**
- Protobuf 33.4_1 detected and linked successfully
- Located at: `/opt/homebrew/opt/protobuf/`
- Required for ONNX model parsing in SOFIE

### Observations

**BLAS Detection Issue:**

**Problem Discovered:**
- ROOT built without optimized BLAS support despite OpenBLAS installed
- CMake output showed: `tmva-cpu:BOOL=OFF` initially (later enabled without optimized BLAS)

**Root Cause Analysis:**
- CMake default: `use_gsl_cblas=ON` prevents searching for optimized BLAS libraries
- GSL not installed and `builtin_gsl=OFF` in configuration
- Result: BLAS search was skipped entirely
- OpenBLAS exists at `/opt/homebrew/opt/openblas/` but was never searched

**Why OpenBLAS Wasn't Auto-Detected:**
1. Homebrew packages OpenBLAS as "keg-only" (not symlinked to standard paths)
2. CMake's BLAS search requires explicit `CMAKE_PREFIX_PATH` or `BLA_VENDOR` specification
3. Unlike Protobuf (which was found automatically), OpenBLAS CMake config exists but wasn't in search path

**Impact:**
- TMVA runs with fallback BLAS implementation (functional but slower)
- For evaluation exercises, acceptable performance for understanding code
- Production use would benefit from rebuilding with: `-DUSE_GSL_CBLAS=OFF -DBLA_VENDOR=Apple` or `-DCMAKE_PREFIX_PATH=/opt/homebrew/opt/openblas`

**Decision:** Proceeded without rebuild to focus on Exercise 2-5 implementation work, as BLAS optimization doesn't affect code understanding or GPU alpaka implementation exercises.

**Compilation Database Limitations:**
- Generated `compile_commands.json` primarily contains LLVM/Clang build units
- ROOT source files not fully represented
- Workaround: Manually configured include paths in `.clangd` for proper code navigation
- clangd successfully indexed 3,147 translation units for code exploration

---

## Exercise 2: Get Familiar with ROOT TMVA Deep Learning Code

### 2.1 TMVA Tutorial Exploration

#### 2.1.1 TMVA_CNN_Classification.C

**Execution:**
```bash
root -l tutorials/machine_learning/TMVA_CNN_Classification.C
```

**Tutorial Purpose:** 
Demonstrates 2D convolutional neural network classification using TMVA's deep learning implementation, with optional Keras backend via PyMVA interface.

**Key Observations:**

1. **ROOT Macro System Understanding:**
   - `.C` files are ROOT macros, not standard C++ programs
   - No `main()` function required - ROOT interpreter (Cling) executes them
   - Doxygen-style comments (`///`, `/***/`) are documentation, not syntax errors
   - VSCode/clangd shows errors in `.C` files (expected - they use ROOT interpreter features)

2. **TMVA Workflow:**
   - DataLoader prepares training/testing datasets
   - Factory pattern for method booking (multiple ML algorithms)
   - DNN method supports multiple backends: CPU, CUDA, PyTorch, Keras
   - Training produces weights and evaluation metrics
   - Web-based GUI (`TBrowser`) displays ROC curves and performance plots

3. **Tutorial Execution:**
   - Ran successfully on CPU backend
   - Generated output files: TMVA.root, dataset splits, weight files
   - Web canvas opened showing classification results
   - Performance metrics: accuracy, loss curves, confusion matrix

**Code Architecture Insights:**
- `TMVA::DataLoader` - handles dataset management and preprocessing
- `TMVA::Factory` - central manager for training/evaluation
- Method booking syntax: `factory->BookMethod(dataloader, TMVA::Types::kDNN, "DNN", options)`
- Backend selection via `Architecture=CPU|GPU|CUDNN|OPENCL`

### 2.2 PyMVA Module Architecture

**Location:** `tmva/pymva/`

**Purpose:** Bridge between Python ML frameworks (Keras/PyTorch) and C++ TMVA/SOFIE

**Key Files Explored:**

#### `RModelParser_PyTorch.cxx` (tmva/pymva/src/)
**Functionality:**
- Parses PyTorch models for TMVA integration
- Converts PyTorch models to ONNX format first (standard intermediate representation)
- ONNX format then processed by SOFIE for C++ code generation
- Workflow: PyTorch → ONNX → SOFIE RModel → Generated C++ inference code

**Key Insights:**
- Uses PyTorch's `torch.onnx.export()` for model conversion
- Extracts model architecture, weights, and operator graph
- Maps PyTorch operations to ONNX operators
- ONNX is universal interchange format (Keras/TensorFlow also export to ONNX)

#### `RModelParser_Keras.cxx` (tmva/pymva/src/)
**Functionality:**
- Similar architecture to PyTorch parser
- Keras/TensorFlow models → ONNX → SOFIE
- Handles layer-by-layer conversion
- Preserves trained weights during conversion

**Python-C++ Interface:**
- Uses `TPython` class for Python interpreter integration
- Python code executed from C++ via `TPython::Exec()`
- Model objects passed between Python and C++ through ROOT's PyROOT bindings

### 2.3 SOFIE Module Architecture

**Key Components:**

#### Core Classes:
- **RModel** (`tmva/sofie/inc/TMVA/RModel.hxx`): Central model representation
  - Stores operator graph (directed acyclic graph of operations)
  - Manages tensors (inputs, outputs, intermediate activations, weights)
  - Generates standalone C++ inference code (`.hxx` files)
  - No runtime dependencies - pure C++ template-based execution

- **ROperator_*** (79+ operator implementations): `tmva/sofie/inc/TMVA/ROperator_*.hxx`
  - Each ONNX operator has dedicated C++ class
  - Examples examined:
    - `ROperator_Sigmoid.hxx` - Sigmoid activation: f(x) = 1 / (1 + exp(-x))
    - `ROperator_Tanh.hxx` - Tanh activation: f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    - `ROperator_Elu.hxx` - ELU activation: f(x) = x if x > 0, else alpha * (exp(x) - 1)
  - Each operator implements:
    - `Initialize()` - setup and validation
    - `Generate()` - C++ code generation for inference
    - Shape inference for output tensors

#### Workflow Understanding:
```
[External Model: PyTorch/Keras/TensorFlow]
           ↓
    [ONNX Export]
           ↓
  [RModelParser_ONNX]
           ↓
[SOFIE RModel (operator graph + tensors)]
           ↓
 [Code Generation: RModel.Generate()]
           ↓
[Generated C++ Header: model.hxx]
           ↓
   [Fast CPU/GPU Inference]
```

**Key Design Decisions:**
1. **No Runtime Overhead**: Generated code is pure C++, no interpreter or framework dependencies
2. **Template-Based**: Heavy use of C++ templates for type-safe tensor operations
3. **Operator Composition**: Complex models built from simple operator primitives
4. **Multi-Backend Support**: CPU (via BLAS), GPU (CUDA via cuDNN), experimental alpaka for heterogeneous computing

#### SOFIE Operator Structure (Pattern Observed):
```cpp
class ROperator_<OpName> {
   std::string fNX, fNY;  // Input/output tensor names
   std::vector<size_t> fShape;  // Tensor dimensions
   std::string fType;  // Data type (float, double, etc.)
   
   void Initialize(RModel& model);  // Validate inputs, infer output shapes
   std::string Generate(std::string OpName);  // Generate C++ inference code
};
```

**Code Generation Output:**
- Produces standalone `.hxx` header file
- Contains class with `infer()` method
- Optimized loops for tensor operations
- Supports batching and multi-dimensional tensors

### 2.4 SOFIE Tutorials (Conceptual Understanding)
Through code exploration understood the following:

#### TMVA_SOFIE_ONNX.C Workflow:
1. Load pre-trained ONNX model file
2. Parse with `RModelParser_ONNX`
3. Generate C++ inference code via `model.Generate()`
4. Compile generated header
5. Run inference and compare with ONNX runtime results
6. Demonstrates correctness and performance of generated code

#### TMVA_SOFIE_Keras.C Workflow:
1. Train Keras model in Python (or load pre-trained)
2. Export to ONNX format
3. Parse with SOFIE
4. Generate C++ code
5. Compare Python vs C++ inference results

#### TMVA_SOFIE_PyTorch.C Workflow:
1. Define PyTorch model
2. Use `torch.onnx.export()` for conversion
3. SOFIE processes ONNX
4. Generated code used for deployment

**Key Insight:** All three workflows converge on ONNX as intermediate representation, demonstrating SOFIE's framework-agnostic design.

### 2.5 Key Learnings from Exercise 2

**1. TMVA's Modular Architecture:**
- Clean separation: data loading, training, evaluation
- Factory pattern enables easy algorithm comparison
- Multiple backend support (CPU, GPU, frameworks)

**2. PyMVA as Integration Layer:**
- Not a new implementation, but a bridge
- Leverages existing frameworks' training capabilities
- Focuses on inference optimization via SOFIE

**3. SOFIE's Code Generation Approach:**
- Compile-time optimization vs runtime interpretation
- Removes framework dependencies for deployment
- Trade-off: larger binary size for faster execution
- Ideal for production inference where models are static

**4. ONNX as Universal Format:**
- Industry standard for model interchange
- Enables framework-agnostic inference
- Well-defined operator specifications
- 79+ operators in SOFIE cover most common DL architectures

**5. Operator Implementation Pattern:**
- Each operator is self-contained C++ class
- Code generation paradigm: operators emit C++ code strings
- Type-safe tensor operations via templates
- Shape inference crucial for memory allocation


---

## AI Usage Declaration

### Transparency Statement
As per CERN HSF guidelines allowing AI usage, I am documenting all AI assistance received during these exercises with complete honesty.

### Specific AI Assistance Areas

#### 1. Development Environment Setup
**Problem:** VSCode showing numerous false errors in ROOT codebase, making code navigation difficult.

**AI Assistance:**
- Diagnosed clangd configuration issues
- Explained why `-fno-exceptions` flag from LLVM build caused crashes
- Guided creation of `.clangd` configuration file
- Helped understand compilation database structure and limitations
- Provided include path configuration for ROOT source tree

**My Work:** 
- Installed clangd independently
- Tested various configurations
- Debugged specific error messages
- Created final `.clangd` and `.vscode/settings.json` files
- Verified clangd indexing success

#### 4. Code Architecture Understanding
**Problem:** Large codebase (ROOT ~2M lines of code), needed guidance on where to focus.

**AI Assistance:**
- Provided high-level SOFIE architecture overview
- Explained PyMVA integration workflow (PyTorch/Keras → ONNX → SOFIE)
- Guided reading of specific files (`RModel.cxx`, operator headers)
- Explained operator implementation pattern
- Clarified ONNX's role as intermediate format

**My Work:**
- Read all mentioned source files independently
- Traced code paths through multiple files
- Examined operator implementations (`ROperator_Sigmoid.hxx`, etc.)
- Built mental model of data flow through system
- Connected tutorial execution to source code implementation

#### 5. Documentation Structure

**AI Assistance:**
- Provided template structure for notes.md
- Suggested sections to include based on exercise requirements
- Recommended documenting known issues (BLAS detection)

**My Work:**
- All technical content written independently based on actual work performed
- Decision-making throughout build and exploration process
- Git commit strategy and messages

### What AI Did NOT Do
- **No code written by AI**: All configuration files (`.clangd`, `.vscode/settings.json`) created by me based on AI guidance
- **No commands executed by AI**: All cmake, build, git, and tutorial commands run manually by me
- **No exercise solutions provided**: AI guided exploration, did not provide answers
