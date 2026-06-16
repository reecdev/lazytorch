# lazytorch
The easiest ML library you'll ever use. **Currently in early beta. Do not use for production yet.**

## What is LazyTorch?
LazyTorch is a ML library that lets you make neural networks and models with ease. Just like the name suggests, LazyTorch is for lazy people like me.

## Installation
```bash
pip install lazytorch
```

## Code Examples
**Brain** - General classification and neural networks.
```python
model = Brain(input={"text": ""}, output={"is_positive": True})

ds = Dataset(
    {"text": [
        "i love this",
        "this is great",
        "absolute perfection",
        "i hate this",
        "this is terrible",
        "this is awful"
    ]},
    {"is_positive": [True, True, True, False, False, False]}
)

model.teach(ds)

print(model.infer({"text": "i love this"})) # outputs: {'is_positive': True}
print(model.infer({"text": "this is terrible"})) # outputs: {'is_positive': False}
```

**AutoregressiveBrain** - Brain but for autoregressive text generation and word prediction.
```python
model = AutoregressiveBrain(context_size=2)
model.teach("the capital of russia is moscow. the capital of britan is london. the capital of france is paris. the capital of greece is athens.")
print(model.generate("the capital of france is", max_words=1)) # output: paris
```

**Dataset** - Datasets.
```python
ds = Dataset({"text": ["hello", "world"]}, {"is_positive": [True, True]})
model = Brain(input={"text": ""}, output={"is_positive": True})
model.teach(ds)
```

**BrainTrust** - Sample results from multiple Brain models to get a general consensus.
```python
ensemble = BrainTrust([model_a, model_b])
print(ensemble.infer({"text": "spam message"})) # output: {'is_spam': True}
```

**BrainScanner** - Evaluate Brain models on datasets.
```python
accuracy = BrainScanner.evaluate(model, ds)
print(accuracy) # output: 1.0 (100%)
```

**These are all of the functions available in the beta of LazyTorch. These are not final.**
