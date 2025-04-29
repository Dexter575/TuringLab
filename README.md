# TuringLab

**TuringLab** is a collection of physics-based discrete-time simulation modules designed for computational scientists, engineers, and researchers. It encompasses numerical solvers, inverse problem frameworks, and optimization tools, all implemented in Python.

---

## 📁 Project Modules

The repository is organized into the following key modules:

- **`accoustic_waves/`** — Simulations of acoustic wave propagation using finite difference methods.
- **`fbp/`** — Implementation of Filtered Back Projection for computed tomography reconstruction.
- **`fourier_optics/`** — Experiments related to Fourier optics and wave propagation.
- **`harmonic_oscillator/`** — Models of harmonic oscillators and their dynamics.
- **`heat_equation/`** — Numerical solutions to the heat equation in various dimensions.
- **`inverse_problems/`** — Frameworks for solving inverse problems in physics.
- **`multi-agent/`** — Simulations involving multiple interacting agents.
- **`optimization/`** — Optimization algorithms applied to physical systems.
- **`solvers/finite_difference/`** — Finite difference solvers for partial differential equations.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have [Conda](https://docs.conda.io/en/latest/) installed.

### Installation

```bash
git clone https://github.com/Dexter575/TuringLab.git
cd TuringLab
```

Create and activate a Conda environment:

```bash
conda create -n turinglab_env python=3.8
conda activate turinglab_env
```

Install the required packages:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, install common dependencies manually:

```bash
pip install numpy scipy matplotlib pint
```

---

## 🧪 Running Experiments

To run experiments, navigate to the respective module directory and execute the desired script.

**Example (from the Fourier Optics module):**

```bash
python fourier_optics/experiment_one.py
python fourier_optics/experiment_two.py
```

> Replace the filenames with the scripts in other modules as needed.

---

## 📚 Documentation

Each module contains its own `README.md` file with:
- Theoretical background
- Experiment explanations
- Usage instructions

Refer to those for detailed module-specific guidance.

---

## 🤝 Contributing

Contributions are welcome!  
If you have suggestions for improvements or new modules, feel free to:
- Fork the repository
- Create a feature branch
- Submit a pull request

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🌐 Repository

Visit the main repository here: [TuringLab on GitHub](https://github.com/Dexter575/TuringLab)


---

## 👨‍🔬 Authors

- **Idrees M. (Dexter575)** — Project Creator and Core Developer  
  [GitHub Profile](https://github.com/Dexter575)
- Contributions are welcome! Feel free to add your name here via Pull Request if you contribute.

