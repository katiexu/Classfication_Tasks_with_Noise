import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
import random
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
import pickle
from tqdm import tqdm

# Qiskit imports
from qiskit import QuantumCircuit, transpile
from qiskit_aer.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.fake_provider import FakeQuito
from qiskit_aer.noise import NoiseModel as AerNoiseModel
from qiskit.utils import algorithm_globals

# Dataset import
from torchquantum.dataset import MNIST


class Qiskit_Quantum_Model(torch.nn.Module):
    def __init__(self, n_qubits, use_noise_model_train=True, use_noise_model_infer=True, backend_device='CPU'):
        super().__init__()
        self.n_qubits = n_qubits
        self.use_noise_model_train = use_noise_model_train
        self.use_noise_model_infer = use_noise_model_infer
        self.backend_device = backend_device

        # Trainable quantum circuit parameters
        self.u3_params = torch.nn.Parameter(torch.randn(n_qubits, 3) * 0.1, requires_grad=True)
        self.cu3_params = torch.nn.Parameter(torch.randn(n_qubits, 3) * 0.1, requires_grad=True)

        # Fully connect layer (classical computing)
        # Input dimension is 4, corresponding to the expectation values of 4 qubits.
        # Output dimension is 2, corresponding to the binary classification task.
        self.fc = torch.nn.Linear(n_qubits, 2)

        # Setup Qiskit noise backend
        self.setup_qiskit_noise_backend()

    def setup_qiskit_noise_backend(self):
        """Setup Qiskit noise backend"""
        # 加载噪声模型，但不立即决定是否使用
        try:
            with open('NoiseModel/my_fake_quito_noise.pkl', 'rb') as file:
                noise_model_dict = pickle.load(file)
            self.noise_model = AerNoiseModel().from_dict(noise_model_dict)
            print("Successfully loaded custom noise model")
        except Exception as e:
            print(f"Error loading noise model: {e}")
            self.noise_model = None

        self.shot = 6000
        self.seeds = 170
        algorithm_globals.random_seed = self.seeds

        # Use FakeQuito backend (5 qubits)
        self.system_model = FakeQuito()

    def create_quantum_circuit(self, x):
        # Preprocess data: downsample, flatten and scale
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)
        x_scaled = x * np.pi

        quantum_circuits = []

        for i in range(bsz):
            qc = QuantumCircuit(self.n_qubits)

            # Part 1: data encoder
            gate_sequence = ['ry'] * self.n_qubits + ['rx'] * self.n_qubits + ['rz'] * self.n_qubits + [
                'ry'] * self.n_qubits

            for j, gate_type in enumerate(gate_sequence):
                qubit_idx = j % self.n_qubits
                angle = float(x_scaled[i, j].detach())

                if gate_type == 'ry':
                    qc.ry(angle, qubit_idx)
                elif gate_type == 'rx':
                    qc.rx(angle, qubit_idx)
                elif gate_type == 'rz':
                    qc.rz(angle, qubit_idx)

            # Part 2: U3 gates (with trainable parameters)
            for qubit in range(self.n_qubits):
                theta = float(self.u3_params[qubit, 0])
                phi = float(self.u3_params[qubit, 1])
                lam = float(self.u3_params[qubit, 2])
                qc.u(theta, phi, lam, qubit)

            # Part 3: CU3 gates (with trainable parameters)
            connections = [(i, (i + 1) % self.n_qubits) for i in range(self.n_qubits)]

            for idx, (control, target) in enumerate(connections):
                theta = float(self.cu3_params[idx, 0])
                phi = float(self.cu3_params[idx, 1])
                lam = float(self.cu3_params[idx, 2])
                qc.cu(theta, phi, lam, 0, control, target)

            quantum_circuits.append(qc)

        return quantum_circuits

    def adjust_observable_for_circuit(self, observable, circuit):
        """Adjust observable based on circuit qubit count"""
        circuit_qubits = circuit.num_qubits
        observable_qubits = len(observable.paulis[0])

        if circuit_qubits == observable_qubits:
            return observable
        elif circuit_qubits > observable_qubits:
            padding = "I" * (circuit_qubits - observable_qubits)
            pauli_strs = [str(pauli) for pauli in observable.paulis]
            adjusted_paulis = [pauli_str + padding for pauli_str in pauli_strs]
            return SparsePauliOp(adjusted_paulis)
        else:
            pauli_strs = [str(pauli) for pauli in observable.paulis]
            adjusted_paulis = [pauli_str[:circuit_qubits] for pauli_str in pauli_strs]
            return SparsePauliOp(adjusted_paulis)

    def create_pauli_observables(self):
        """Create Pauli-Z observables for each qubit"""
        observables = []
        for qubit in range(self.n_qubits):
            # Create observable string like "ZIII", "IZII", etc.
            pauli_str = "I" * qubit + "Z" + "I" * (self.n_qubits - qubit - 1)
            observable = SparsePauliOp.from_list([(pauli_str, 1.0)])
            observables.append(observable)
        return observables

    def run_qiskit_simulator(self, quantum_circuits, observables, is_training=True):
        results = []

        # 根据是训练还是推断阶段决定是否使用噪声
        if is_training:
            use_noise = self.use_noise_model_train
            phase = "training"
        else:
            use_noise = self.use_noise_model_infer
            phase = "inference"

        # 只在第一个batch时打印，避免重复输出
        if not hasattr(self, f'_printed_{phase}'):
            print(f"Running quantum simulation for {phase} phase - Noise: {use_noise}")
            setattr(self, f'_printed_{phase}', True)

        for i, qc in enumerate(quantum_circuits):
            try:
                transpiled_qc = transpile(qc, backend=self.system_model, optimization_level=1)
            except Exception as e:
                print(f"Error transpiling circuit {i}: {e}")
                transpiled_qc = qc

            circuit_results = []
            for observable in observables:
                try:
                    adjusted_observable = self.adjust_observable_for_circuit(observable, transpiled_qc)

                    backend_options = {
                        'method': 'statevector',
                        'device': self.backend_device,
                    }
                    if use_noise and self.noise_model is not None:
                        backend_options['noise_model'] = self.noise_model

                    estimator = Estimator(
                        backend_options=backend_options,
                        run_options={
                            'shots': self.shot,
                            'seed': self.seeds,
                        },
                        skip_transpilation=True
                    )

                    job = estimator.run(transpiled_qc, adjusted_observable)
                    result = job.result()
                    expectation_value = result.values[0]
                    circuit_results.append(expectation_value)

                except Exception as e:
                    print(f"Error running quantum circuit {i} for observable: {e}")
                    circuit_results.append(0.0)

            results.append(circuit_results)

        return torch.tensor(results, dtype=torch.float32, requires_grad=True)

    def forward(self, x, is_training=True):
        device = x.device

        # Create quantum circuits
        quantum_circuits = self.create_quantum_circuit(x)

        # Create Pauli-Z observables for each qubit
        observables = self.create_pauli_observables()

        # Run qiskit simulator with phase information
        quantum_results = self.run_qiskit_simulator(quantum_circuits, observables, is_training=is_training)
        quantum_results = quantum_results.to(device)

        # Ensure results require gradients
        if not quantum_results.requires_grad:
            quantum_results.requires_grad_(True)

        # Fully connect layer (classical computing)
        classical_output = self.fc(quantum_results)

        # Apply log softmax
        output = F.log_softmax(classical_output, dim=1)

        return output


def train(dataflow, model, device, optimizer):
    # 重置打印标志
    if hasattr(model, '_printed_training'):
        delattr(model, '_printed_training')

    pbar = tqdm(dataflow["train"], desc="Training", unit="batch")

    total_loss = 0.0
    for batch_idx, feed_dict in enumerate(pbar):
        inputs = feed_dict["image"].to(device)
        targets = feed_dict["digit"].to(device)

        # 训练阶段传入 is_training=True
        outputs = model(inputs, is_training=True)
        loss = F.nll_loss(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)

        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg Loss': f'{avg_loss:.4f}'
        })


def valid_test(dataflow, split, model, device):
    # 重置打印标志
    if hasattr(model, '_printed_inference'):
        delattr(model, '_printed_inference')

    target_all = []
    output_all = []

    with torch.no_grad():
        pbar = tqdm(dataflow[split], desc=f"{split.capitalize()}ing", unit="batch")
        for feed_dict in pbar:
            inputs = feed_dict["image"].to(device)
            targets = feed_dict["digit"].to(device)

            # 推断阶段传入 is_training=False
            outputs = model(inputs, is_training=False)

            target_all.append(targets)
            output_all.append(outputs)

        target_all = torch.cat(target_all, dim=0)
        output_all = torch.cat(output_all, dim=0)

    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size
    loss = F.nll_loss(output_all, target_all).item()

    print(f"{split} set accuracy: {accuracy:.4f}")
    print(f"{split} set loss: {loss:.4f}")

    return accuracy, loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5, help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--n-qubits", type=int, default=4,
                        help="number of qubits for quantum circuit")

    # 替换原来的 --no-noise 参数为两个独立的参数
    parser.add_argument("--noise-train", action="store_true", default=False,
                        help="enable noise model during training")
    parser.add_argument("--noise-infer", action="store_true", default=True,
                        help="enable noise model during inference")

    parser.add_argument("--pdb", action="store_true", help="debug with pdb")

    # Add new parameters
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="device for torch (cuda/cpu)")
    parser.add_argument("--lr", type=float, default=5e-3, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--backend-device", type=str, default="CPU", choices=["CPU", "GPU"],
                        help="device for Qiskit backend (CPU/GPU)")

    args = parser.parse_args()

    if args.pdb:
        import pdb
        pdb.set_trace()

    # Set random seeds
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load dataset
    dataset = MNIST(
        root="./mnist_data",
        train_valid_split_ratio=[0.9, 0.1],
        digits_of_interest=[3, 6],
    )

    dataflow = dict()
    for split in dataset:
        sampler = torch.utils.data.RandomSampler(dataset[split])
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=False,
        )

    device = torch.device(args.device)

    print(f"Training configuration:")
    print(f"  Noise configuration: train={args.noise_train}, inference={args.noise_infer}")
    print(f"  Using device: {device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Number of qubits: {args.n_qubits}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Noise during training: {args.noise_train}")
    print(f"  Noise during inference: {args.noise_infer}")
    print(f"  Torch device: {args.device}")
    print(f"  Qiskit backend device: {args.backend_device}")

    model = Qiskit_Quantum_Model(
        n_qubits=args.n_qubits,
        use_noise_model_train=args.noise_train,
        use_noise_model_infer=args.noise_infer,
        backend_device=args.backend_device
    ).to(device)

    print(f"\nModel structure:")
    print(f"  Quantum parameters: U3 ({model.u3_params.shape}), CU3 ({model.cu3_params.shape})")
    print(f"  Classical layer: Linear({model.fc.in_features} -> {model.fc.out_features})")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total trainable parameters: {total_params}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    accuracy_list = []
    loss_list = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}:")

        # Training phase
        train(dataflow, model, device, optimizer)

        # Validation phase
        accuracy, loss = valid_test(dataflow, "valid", model, device)
        accuracy_list.append(accuracy)
        loss_list.append(loss)

        scheduler.step()

    # Testing phase
    print("\nFinal Testing:")
    valid_test(dataflow, "test", model, device)


if __name__ == "__main__":
    main()