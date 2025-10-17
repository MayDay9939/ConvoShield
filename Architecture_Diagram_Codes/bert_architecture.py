from graphviz import Digraph

def visualize_bert():
    dot = Digraph(format='png')
    
    # Input
    dot.node('Input', 'Input Text', shape='parallelogram', style='filled', fillcolor='lightblue')
    dot.node('Tokenization', 'Tokenization', shape='box', style='filled', fillcolor='lightgray')
    dot.edge('Input', 'Tokenization')
    
    # Embeddings
    dot.node('Embeddings', 'Word + Positional + Token Type Embeddings', shape='box', style='filled', fillcolor='lightyellow')
    dot.edge('Tokenization', 'Embeddings')
    
    # Encoder Block
    dot.node('Encoder', 'BERT Encoder (6 Layers)', shape='box', style='filled', fillcolor='lightcoral')
    dot.edge('Embeddings', 'Encoder')
    
    # Attention
    dot.node('Self-Attn', 'Self-Attention (Q, K, V, Softmax)', shape='box', style='filled', fillcolor='lightpink')
    dot.node('FFN', 'Feed-Forward Network (GELU)', shape='box', style='filled', fillcolor='lightpink')
    dot.node('NormDropout', 'LayerNorm + Dropout', shape='box', style='filled', fillcolor='lightpink')
    
    dot.edge('Encoder', 'Self-Attn')
    dot.edge('Self-Attn', 'FFN')
    dot.edge('FFN', 'NormDropout')
    dot.edge('NormDropout', 'Encoder')
    
    # Pooling Layer
    dot.node('Pooler', 'Pooler Layer (Tanh)', shape='box', style='filled', fillcolor='lightgreen')
    dot.edge('Encoder', 'Pooler')
    
    # Dropout
    dot.node('Dropout', 'Dropout Layer', shape='box', style='filled', fillcolor='lightgray')
    dot.edge('Pooler', 'Dropout')
    
    # Classifier
    dot.node('Classifier', 'Classifier (Linear Layer)', shape='box', style='filled', fillcolor='lightblue')
    dot.edge('Dropout', 'Classifier')
    
    # Output
    dot.node('Output', 'Prediction', shape='parallelogram', style='filled', fillcolor='lightblue')
    dot.edge('Classifier', 'Output')
    
    return dot

# Generate and render the diagram
dot = visualize_bert()
dot.render('bert_architecture', format='png', view=True)
