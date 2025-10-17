import graphviz

def draw_gptneo_model():
    dot = graphviz.Digraph('GPTNeoForSequenceClassification', format='png')
    dot.attr(size='12,12', rankdir='TB', splines='ortho')  # Adjust size and layout for better clarity
    
    # Main model
    dot.node('Model', 'GPTNeoForSequenceClassification', shape='box', style='filled', fillcolor='lightblue', fontsize='14', width='3')
    
    # Transformer block
    dot.node('Transformer', 'GPTNeoModel', shape='box', style='filled', fillcolor='lightgrey', fontsize='14', width='2.5')
    dot.edge('Model', 'Transformer')
    
    # Embedding layers
    dot.node('wte', 'Word Embedding\n(50257, 768)', shape='ellipse', width='2')
    dot.node('wpe', 'Position Embedding\n(2048, 768)', shape='ellipse', width='2')
    dot.node('drop', 'Dropout\n(p=0.0)', shape='ellipse', width='1.5')
    dot.edge('Transformer', 'wte')
    dot.edge('Transformer', 'wpe')
    dot.edge('Transformer', 'drop')

    # Multi-layer transformer blocks (Collapsed representation)
    with dot.subgraph() as sub:
        sub.attr(label="Repeated 12 times", style="dashed", fontsize='12')
        sub.node('h', 'GPTNeoBlock x12', shape='box', style='filled', fillcolor='lightgrey', fontsize='14', width='2.5')
        dot.edge('Transformer', 'h')

    # Expanded view of one GPTNeoBlock for clarity
    dot.node('Block', 'GPTNeoBlock', shape='box', style='filled', fillcolor='lightyellow', fontsize='12', width='2')
    dot.edge('h', 'Block', style="dashed", label="Example structure of one block")

    # LayerNorm, Attention, and MLP in a single block
    dot.node('ln1', 'LayerNorm\n(768)', shape='ellipse', width='1.5')
    dot.node('attn', 'GPTNeoAttention', shape='ellipse', width='1.8')
    dot.node('ln2', 'LayerNorm\n(768)', shape='ellipse', width='1.5')
    dot.node('mlp', 'GPTNeoMLP', shape='ellipse', width='1.8')

    dot.edge('Block', 'ln1')
    dot.edge('Block', 'attn')
    dot.edge('Block', 'ln2')
    dot.edge('Block', 'mlp')

    # Final LayerNorm
    dot.node('ln_f', 'LayerNorm\n(768)', shape='ellipse', width='2')
    dot.edge('h', 'ln_f')

    # Output layer
    dot.node('score', 'Linear\n(768 → 2)', shape='ellipse', style='filled', fillcolor='lightgreen', width='2')
    dot.edge('ln_f', 'score')
    dot.edge('score', 'Model', label='Output')

    # Render the diagram
    dot.render('gptneo_model_architecture_updated', view=True)

# Draw and visualize the model
if __name__ == "__main__":
    draw_gptneo_model()
