import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
import random
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Qiskit imports
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.fake_provider import FakeQuito
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import Estimator
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

        # Setup Qiskit noise backend
        self.setup_qiskit_noise_backend()

    def setup_qiskit_noise_backend(self):
        """Setup Qiskit noise backend"""
        try:
            # Build noise model from backend properties
            self.backend = FakeQuito()      # Select the qiskit backend to use
            self.noise_model = NoiseModel.from_backend(self.backend)
            print("✅ Successfully loading noise model from Qiskit backend")
        except Exception as e:
            print(f"❌ Error loading noise model from Qiskit backend: {e}")

    def create_quantum_circuit(self, x):
        # Preprocess data: downsample, flatten and scale
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)
        # x_scaled = x * np.pi

        quantum_circuits = []

        for i in range(bsz):
            qc = QuantumCircuit(self.n_qubits)

            # Part 1: data encoder
            gate_sequence = ['ry'] * self.n_qubits + ['rx'] * self.n_qubits + ['rz'] * self.n_qubits + [
                'ry'] * self.n_qubits

            for j, gate_type in enumerate(gate_sequence):
                qubit_idx = j % self.n_qubits
                angle = float(x[i, j].detach())

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

    def create_pauli_observable(self):
        """Create a combined Pauli-Z observable for all qubits"""
        # Create Z observable for each qubit and combine into a comprehensive observable
        observables = []
        for qubit in range(self.n_qubits):
            # Create observable string like "ZIII", "IZII", etc.
            pauli_str = "I" * qubit + "Z" + "I" * (self.n_qubits - qubit - 1)
            observable = SparsePauliOp.from_list([(pauli_str, 1.0)])
            observables.append(observable)

        # Sum all observables to create a combined observable
        combined_observable = observables[0]
        for obs in observables[1:]:
            combined_observable = combined_observable + obs

        return combined_observable

    def adjust_observable_for_circuit(self, observable, circuit):
        """Adjust observable based on transpiled circuit qubit count"""
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

    def run_qiskit_simulator(self, quantum_circuits, observable, is_training=True):
        backend_seeds = 170
        algorithm_globals.random_seed = backend_seeds
        seed_transpiler = backend_seeds
        shot = 6000

        # Decide whether to apply noise based on training or inference phase
        if is_training:
            use_noise = self.use_noise_model_train
            phase = "training"
        else:
            use_noise = self.use_noise_model_infer
            phase = "inference"

        if not hasattr(self, f'_printed_{phase}'):
            print(f"Running quantum simulation for {phase} phase - Noise: {use_noise}")
            setattr(self, f'_printed_{phase}', True)

        if use_noise:
            estimator = Estimator(
                backend_options={
                    'method': 'statevector',
                    'device': self.backend_device,
                    'noise_model': self.noise_model  # Add noise model when noise is enabled
                },
                run_options={
                    'shots': shot,
                    'seed': backend_seeds,
                },
                transpile_options={
                    'seed_transpiler': seed_transpiler
                }
            )
        else:
            estimator = Estimator(
                backend_options={
                    'method': 'statevector',
                    'device': self.backend_device,
                    # Do not use noise model when noise is disabled
                },
                run_options={
                    'shots': shot,
                    'seed': backend_seeds,
                },
                transpile_options={
                    'seed_transpiler': seed_transpiler
                }
            )

        results = []
        for i, qc in enumerate(quantum_circuits):
            transpiled_qc = transpile(qc, backend=self.backend)
            adjusted_observable = self.adjust_observable_for_circuit(observable, transpiled_qc)
            try:
                job = estimator.run(transpiled_qc, adjusted_observable)
                result = job.result()
                expectation_value = result.values[0]

                # Handle binary classification: use expectation value as logits
                quantum_output = torch.tensor([[expectation_value, -expectation_value]], dtype=torch.float32)
                results.append(quantum_output)

            except Exception as e:
                print(f"Error running quantum circuit {i}: {e}")
                # Use default value [0.5, -0.5] when error occurs
                default_output = torch.tensor([[0.5, -0.5]], dtype=torch.float32)
                results.append(default_output)

        # Stack results into shape [batch_size, 2]
        if results:
            quantum_results = torch.cat(results, dim=0)
        else:
            # Create default output if no results
            quantum_results = torch.zeros((len(quantum_circuits), 2), dtype=torch.float32)
            quantum_results[:, 0] = 0.5  # class 3
            quantum_results[:, 1] = -0.5  # class 6

        return quantum_results


    def forward(self, x, is_training=True):
        device = x.device

        # Create quantum circuits
        quantum_circuits = self.create_quantum_circuit(x)

        # Create a combined Pauli-Z observable
        observable = self.create_pauli_observable()

        # Run qiskit simulator with phase information
        quantum_results = self.run_qiskit_simulator(quantum_circuits, observable, is_training=is_training)
        quantum_results = quantum_results.to(device)

        # Ensure results require gradients
        if not quantum_results.requires_grad:
            quantum_results.requires_grad_(True)

        output = F.log_softmax(quantum_results, dim=1)

        return output


def train(dataflow, model, device, optimizer):
    if hasattr(model, '_printed_training'):
        delattr(model, '_printed_training')

    pbar = tqdm(dataflow["train"], desc="Training", unit="batch")

    total_loss = 0.0
    for batch_idx, feed_dict in enumerate(pbar):
        inputs = feed_dict["image"].to(device)
        targets = feed_dict["digit"].to(device)

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
    if hasattr(model, '_printed_inference'):
        delattr(model, '_printed_inference')

    target_all = []
    output_all = []
    input_samples = []

    with torch.no_grad():
        pbar = tqdm(dataflow[split], desc=f"{split.capitalize()}ing", unit="batch")
        for feed_dict in pbar:
            inputs = feed_dict["image"].to(device)
            targets = feed_dict["digit"].to(device)

            outputs = model(inputs, is_training=False)

            target_all.append(targets)
            output_all.append(outputs)
            # Save first few samples for detailed analysis
            if len(input_samples) < 10:  # Only save first 10 samples for analysis
                # Save the first sample in the batch
                input_samples.append((inputs[0:1].cpu(), targets[0:1].cpu(), outputs[0:1].cpu()))

        target_all = torch.cat(target_all, dim=0)
        output_all = torch.cat(output_all, dim=0)

    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size
    loss = F.nll_loss(output_all, target_all).item()

    print(f"\n{split.upper()} SET RESULTS:")
    print(f"=" * 50)
    print(f"Overall Accuracy: {accuracy:.4f} ({corrects}/{size})")
    print(f"Overall Loss: {loss:.4f}")

    # Detailed output of classification process for first few samples
    print(f"\nDETAILED SAMPLE ANALYSIS (first {len(input_samples)} samples):")
    print(f"=" * 50)

    for i, (inputs, targets, outputs) in enumerate(input_samples):
        # Convert log softmax back to probabilities
        probs = torch.exp(outputs)

        print(f"\nSample {i + 1}:")
        print(f"  True label: {targets[0].item()} ({'3' if targets[0].item() == 0 else '6'})")
        print(f"  Raw log probabilities: [{outputs[0][0]:.4f}, {outputs[0][1]:.4f}]")
        print(f"  Probabilities: [3: {probs[0][0]:.4f}, 6: {probs[0][1]:.4f}]")

        # Classification decision process
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = torch.max(probs, dim=1).values.item()

        print(f"  Prediction: {pred_class} ({'3' if pred_class == 0 else '6'})")
        print(f"  Confidence: {confidence:.4f}")
        print(f"  Correct: {'✓' if pred_class == targets[0].item() else '✗'}")

        # Comparison analysis
        prob_3, prob_6 = probs[0][0].item(), probs[0][1].item()
        diff = abs(prob_3 - prob_6)
        larger_prob = "3" if prob_3 > prob_6 else "6"

        print(f"  Comparison: P(3)={prob_3:.4f} vs P(6)={prob_6:.4f}")
        print(f"  Difference: {diff:.4f} (larger: {larger_prob})")
        print(f"  Decision: Choose {larger_prob} because P({larger_prob}) > P({'6' if larger_prob == '3' else '3'})")

    # Statistical confidence distribution
    all_probs = torch.exp(output_all)
    confidences = torch.max(all_probs, dim=1).values
    avg_confidence = confidences.mean().item()

    print(f"\nCONFIDENCE ANALYSIS:")
    print(f"  Average confidence: {avg_confidence:.4f}")
    print(f"  Min confidence: {confidences.min().item():.4f}")
    print(f"  Max confidence: {confidences.max().item():.4f}")

    # Statistical samples close to decision boundary
    close_calls = (confidences < 0.6).sum().item()
    print(f"  Samples with confidence < 0.6: {close_calls}/{size} ({close_calls / size:.2%})")

    return accuracy, loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1, help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--n-qubits", type=int, default=4,
                        help="number of qubits for quantum circuit")

    parser.add_argument("--noise-train", action="store_true", default=False,
                        help="enable noise model during training")      # Set False if not using noise model in training
    parser.add_argument("--noise-infer", action="store_true", default=True,
                        help="enable noise model during inference")     # Set True if using noise model in valid and test

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
        # n_train_samples=50,
        # n_valid_samples=50,
        # n_test_samples=50,
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

    print(f"TRAINING CONFIGURATION:")
    print(f"=" * 50)
    print(f"  Noise configuration: train={args.noise_train}, inference={args.noise_infer}")
    print(f"  Using device: {device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Number of qubits: {args.n_qubits}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Digits: 3 vs 6")

    model = Qiskit_Quantum_Model(
        n_qubits=args.n_qubits,
        use_noise_model_train=args.noise_train,
        use_noise_model_infer=args.noise_infer,
        backend_device=args.backend_device
    ).to(device)

    print(f"\nMODEL STRUCTURE:")
    print(f"  Quantum parameters: U3 ({model.u3_params.shape}), CU3 ({model.cu3_params.shape})")
    print(f"  No classical layer - using direct quantum measurements")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total trainable parameters: {total_params}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    accuracy_list = []
    loss_list = []

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"EPOCH {epoch}/{args.epochs}")
        print(f"{'=' * 60}")

        # Training phase
        train(dataflow, model, device, optimizer)

        # Validation phase
        accuracy, loss = valid_test(dataflow, "valid", model, device)
        accuracy_list.append(accuracy)
        loss_list.append(loss)

        scheduler.step()

    # Testing phase
    print(f"\n{'=' * 60}")
    print(f"FINAL TESTING")
    print(f"{'=' * 60}")
    valid_test(dataflow, "test", model, device)


if __name__ == "__main__":
    main()