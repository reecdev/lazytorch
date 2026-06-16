import torch
import torch.nn as nn
import torch.optim as optim
import hashlib
import random

class AutoAccelerator:
    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def parallelize(model):
        if torch.cuda.device_count() > 1:
            return nn.DataParallel(model)
        return model

class Dataset:
    def __init__(self, input_dict: dict, output_dict: dict = None):
        self.inputs = input_dict
        self.outputs = output_dict if output_dict is not None else {}
        self.keys_in = list(self.inputs.keys())
        self.keys_out = list(self.outputs.keys())
        
        first_val = self.inputs[self.keys_in[0]]
        self.length = len(first_val) if isinstance(first_val, list) else 1
        
        if self.length == 1:
            for k in self.keys_in:
                if not isinstance(self.inputs[k], list):
                    self.inputs[k] = [self.inputs[k]]
            for k in self.keys_out:
                if not isinstance(self.outputs[k], list):
                    self.outputs[k] = [self.outputs[k]]

    def __len__(self):
        return self.length

    def get_batch(self):
        return self.inputs, self.outputs

class MemoryBuffer:
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.x_data = []
        self.y_data = []

    def add(self, x: torch.Tensor, y: torch.Tensor):
        self.x_data.append(x)
        self.y_data.append(y)
        if len(self.x_data) > self.max_size:
            self.x_data.pop(0)
            self.y_data.pop(0)

    def get_all(self):
        if not self.x_data:
            return None, None
        return torch.stack(self.x_data), torch.stack(self.y_data)

class Brain:
    def __init__(self, input: dict, output: dict, num_layers: int = 1, learning_rate: float = 0.01):
        self.device = AutoAccelerator.get_device()
        self.input_dim = 256
        self.output_keys = list(output.keys())
        self.output_dim = len(self.output_keys)
        
        self.output_types = {k: isinstance(v, bool) for k, v in output.items()}

        if any(isinstance(v, str) for v in output.values()):
            raise ValueError("String outputs are not supported in regular brains! If you absolutely have to generate text, consider using AutoregressiveBrain")
        
        layers = [nn.Linear(self.input_dim, 128), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(128, 128), nn.ReLU()])
        
        layers.extend([nn.Linear(128, self.output_dim)])
        
        self.model = nn.Sequential(*layers).to(self.device)
        self.model = AutoAccelerator.parallelize(self.model)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.memory = MemoryBuffer()

    def _hash(self, text: str) -> int:
        return int(hashlib.md5(text.encode()).hexdigest(), 16) % self.input_dim

    def _encode_input(self, input_dict: dict) -> torch.Tensor:
        vec = torch.zeros(self.input_dim)
        for k, v in input_dict.items():
            if isinstance(v, str):
                words = v.split()
                if not words:
                    vec[self._hash(f"{k}_empty")] += 1.0
                for word in words:
                    vec[self._hash(f"{k}_{word}")] += 1.0
            elif isinstance(v, (int, float, bool)):
                vec[self._hash(str(k))] += float(v)
        return vec.to(self.device)

    def _encode_output(self, output_dict: dict) -> torch.Tensor:
        vec = torch.zeros(self.output_dim)
        for i, k in enumerate(self.output_keys):
            vec[i] = float(output_dict.get(k, 0.0))
        return vec.to(self.device)

    def teach(self, input, output=None, epochs: int = 5, show_loss: bool = False):
        if isinstance(input, Dataset):
            inputs, outputs = input.get_batch()
            batch_size = len(input)
        else:
            inputs, outputs = input, output
            batch_size = 1
            for v in inputs.values():
                if isinstance(v, list):
                    batch_size = len(v)
                    break

        for i in range(batch_size):
            single_in = {k: (v[i] if isinstance(v, list) else v) for k, v in inputs.items()}
            single_out = {k: (v[i] if isinstance(v, list) else v) for k, v in outputs.items()}
            self.memory.add(self._encode_input(single_in), self._encode_output(single_out))
        
        X, Y = self.memory.get_all()
        if X is None: return

        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            preds = self.model(X)
            loss = self.criterion(preds, Y)
            loss.backward()
            self.optimizer.step()
            
            if show_loss:
                print(f"Epoch {epoch}: {loss.item():.3f}")

    def infer(self, input: dict) -> dict:
        self.model.eval()
        with torch.no_grad():
            x = self._encode_input(input).unsqueeze(0)
            preds = self.model(x).squeeze(0)
            
        return {
            k: bool(preds[i].item() >= 0.5) if self.output_types[k] else preds[i].item() 
            for i, k in enumerate(self.output_keys)
        }

class AutoregressiveBrain:
    def __init__(self, context_size: int = 5, num_layers: int = 1, learning_rate: float = 0.01):
        self.device = AutoAccelerator.get_device()
        self.vocab = {"<PAD>": 0}
        self.inverse_vocab = {0: "<PAD>"}
        self.context_size = context_size
        
        layers = [
            nn.Embedding(10000, 64),
            nn.Flatten(),
            nn.Linear(64 * context_size, 128),
            nn.ReLU()
        ]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(128, 128), nn.ReLU()])
        layers.append(nn.Linear(128, 10000))

        self.model = nn.Sequential(*layers).to(self.device)
        self.model = AutoAccelerator.parallelize(self.model)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def _tokenize(self, text: str):
        tokens = []
        for word in text.split():
            if word not in self.vocab:
                if len(self.vocab) < 10000:
                    idx = len(self.vocab)
                    self.vocab[word] = idx
                    self.inverse_vocab[idx] = word
            tokens.append(self.vocab.get(word, 0))
        return tokens

    def teach(self, text: str, epochs: int = 5, show_loss: bool = False):
        tokens = self._tokenize(text)
        if len(tokens) <= self.context_size: return
        
        X, Y = [], []
        for i in range(len(tokens) - self.context_size):
            X.append(tokens[i:i+self.context_size])
            Y.append(tokens[i+self.context_size])
            
        x_t = torch.tensor(X, dtype=torch.long).to(self.device)
        y_t = torch.tensor(Y, dtype=torch.long).to(self.device)
        
        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            preds = self.model(x_t)
            loss = self.criterion(preds, y_t)
            loss.backward()
            self.optimizer.step()
            
            if show_loss:
                print(f"Epoch {epoch}: {loss.item():.3f}")

    def generate(self, prompt: str, max_words: int = 5) -> str:
        self.model.eval()
        tokens = self._tokenize(prompt)
        result = tokens.copy()
        
        with torch.no_grad():
            for _ in range(max_words):
                ctx = result[-self.context_size:]
                if len(ctx) < self.context_size:
                    ctx = [0] * (self.context_size - len(ctx)) + ctx
                x_t = torch.tensor([ctx], dtype=torch.long).to(self.device)
                preds = self.model(x_t)
                next_token = torch.argmax(preds, dim=1).item()
                result.append(next_token)
                
        return " ".join([self.inverse_vocab.get(t, "") for t in result])

# class VisionBrain(Brain):
#     def __init__(self, output: dict, num_layers: int = 1, learning_rate: float = 0.01):
#         self.device = AutoAccelerator.get_device()
#         self.output_keys = list(output.keys())
#         self.output_dim = len(self.output_keys)
        
#         layers = [
#             nn.Conv2d(3, 16, 3, 1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Flatten()
#         ]
        
#         if num_layers == 1:
#             layers.extend([nn.LazyLinear(self.output_dim), nn.Sigmoid()])
#         else:
#             layers.extend([nn.LazyLinear(128), nn.ReLU()])
#             for _ in range(num_layers - 2):
#                 layers.extend([nn.Linear(128, 128), nn.ReLU()])
#             layers.extend([nn.Linear(128, self.output_dim), nn.Sigmoid()])
            
#         self.model = nn.Sequential(*layers).to(self.device)
#         self.model = AutoAccelerator.parallelize(self.model)
#         self.criterion = nn.BCELoss()
#         self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
#         self.memory = MemoryBuffer()

#     def teach(self, image_tensor: torch.Tensor, output: dict, epochs: int = 5, show_loss: bool = False):
#         y = self._encode_output(output)
#         self.memory.add(image_tensor.to(self.device), y)
#         X, Y = self.memory.get_all()
        
#         self.model.train()
#         for epoch in range(epochs):
#             self.optimizer.zero_grad()
#             preds = self.model(X)
#             loss = self.criterion(preds, Y)
#             loss.backward()
#             self.optimizer.step()
            
#             if show_loss:
#                 print(f"Epoch {epoch}: {loss.item():.3f}")

#     def infer(self, image_tensor: torch.Tensor) -> dict:
#         self.model.eval()
#         with torch.no_grad():
#             preds = self.model(image_tensor.unsqueeze(0).to(self.device)).squeeze(0)
#         return {k: bool(preds[i].item() >= 0.5) for i, k in enumerate(self.output_keys)}

class BrainScanner:
    @staticmethod
    def evaluate(brain: Brain, test_dataset: Dataset) -> float:
        correct = 0
        total = len(test_dataset)
        inputs, outputs = test_dataset.get_batch()
        
        for i in range(total):
            single_in = {k: v[i] for k, v in inputs.items()}
            single_out = {k: v[i] for k, v in outputs.items()}
            pred = brain.infer(single_in)
            
            correct += sum(pred[k] == single_out[k] for k in single_out)
        return correct / total

class BrainTrust:
    def __init__(self, brains: list):
        self.brains = brains

    def infer(self, input: dict) -> dict:
        results = [b.infer(input) for b in self.brains]
        final_result = {}
        
        for k in results[0].keys():
            if isinstance(results[0][k], bool):
                trues = sum(1 for r in results if r[k])
                final_result[k] = trues > (len(self.brains) / 2)
            else:
                total = sum(r[k] for r in results)
                avg = total / len(self.brains)
                final_result[k] = round(avg) 
                
        return final_result

class BrainSurgeon:
    @staticmethod
    def lobotomy(brain: Brain):
        for param in brain.model.parameters():
            param.requires_grad = False
        return brain
