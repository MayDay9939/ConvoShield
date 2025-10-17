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
    
    # Multi-layer transformer blocks
    dot.node('h', '12 x GPTNeoBlock', shape='box', style='filled', fillcolor='lightgrey', fontsize='14', width='2.5')
    dot.edge('Transformer', 'h')
    
    # Transformer layers
    with dot.subgraph(name='cluster_blocks') as sub:
        sub.attr(label='Transformer Blocks', style='dashed', fontsize='14')
        for i in range(12):
            block_name = f'Block_{i}'
            sub.node(block_name, f'GPTNeoBlock {i}', shape='box', style='filled', fillcolor='lightyellow', fontsize='12', width='2')
            sub.edge('h', block_name)
            
            # LayerNorm, Attention, and MLP in each block
            sub.node(f'ln1_{i}', 'LayerNorm\n(768)', shape='ellipse', width='1.5')
            sub.node(f'attn_{i}', 'GPTNeoAttention', shape='ellipse', width='1.8')
            sub.node(f'ln2_{i}', 'LayerNorm\n(768)', shape='ellipse', width='1.5')
            sub.node(f'mlp_{i}', 'GPTNeoMLP', shape='ellipse', width='1.8')
            
            sub.edge(block_name, f'ln1_{i}')
            sub.edge(block_name, f'attn_{i}')
            sub.edge(block_name, f'ln2_{i}')
            sub.edge(block_name, f'mlp_{i}')
    
    # Final LayerNorm
    dot.node('ln_f', 'LayerNorm\n(768)', shape='ellipse', width='2')
    dot.edge('Transformer', 'ln_f')
    
    # Output layer
    dot.node('score', 'Linear\n(768 → 2)', shape='ellipse', style='filled', fillcolor='lightgreen', width='2')
    dot.edge('ln_f', 'score')
    dot.edge('score', 'Model', label='Output')
    
    # Render the diagram
    dot.render('gptneo_model_architecture', view=True)

# Draw and visualize the model
if __name__ == "__main__":
    draw_gptneo_model()
