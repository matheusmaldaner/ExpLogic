# Base import 
import torch 
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
import numpy as np 
from tqdm import tqdm 
import random
import matplotlib.pyplot as plt 
import networkx as nx 
import pandas as pd
import copy 

# Dataset imports
from utils import mnist_dataset

ALL_OPERATIONS = [
    "zero", "and", "not_implies", "a", "not_implied_by", "b", "xor", "or", 
    "not_or", "not_xor", "not_b", "implied_by", "not_a", "implies", "not_and", "one"]

a_not_list = ["not_implied_by", "not_or","not_xor","not_a","implies","not_and"]
b_not_list = ["not_implies", "not_or","not_xor","not_b","implied_by","not_and"]

def calculate_prob_z(learned_gate, probs):
    """
    Calculate the probability Z based on the learned gate and input probabilities prob_a and prob_b.
    
    Args:
    learned_gate (str): The logic gate to apply (from ALL_OPERATIONS).
    prob_a (float): The probability of input A.
    prob_b (float): The probability of input B.
    
    Returns:
    prob_z (float): The resulting probability based on the gate logic.
    """
    
    if learned_gate == 'zero':  # False
        return 0
    elif learned_gate == 'and':  # A ∧ B
        return probs['a'] * probs['b']
    elif learned_gate == 'not_implies':  # ¬(A ⇒ B) = A - A*B
        return probs['a'] - probs['a'] * probs['b']
    elif learned_gate == 'a':  # A
        return probs['a']
    elif learned_gate == 'not_implied_by':  # ¬(A ⇐ B) = B - A*B
        return probs['b'] - probs['a'] * probs['b']
    elif learned_gate == 'b':  # B
        return probs['b']
    elif learned_gate == 'xor':  # A ⊕ B = A + B - 2*A*B
        return probs['a'] + probs['b'] - 2 * probs['a'] * probs['b']
    elif learned_gate == 'or':  # A ∨ B = A + B - A*B
        return probs['a'] + probs['b'] - probs['a'] * probs['b']
    elif learned_gate == 'not_or':  # ¬(A ∨ B) = 1 - (A + B - A*B)
        return 1 - (probs['a'] + probs['b'] - probs['a'] * probs['b'])
    elif learned_gate == 'not_xor':  # ¬(A ⊕ B) = 1 - (A + B - 2*A*B)
        return 1 - (probs['a'] + probs['b'] - 2 * probs['a'] * probs['b'])
    elif learned_gate == 'not_b':  # ¬B = 1 - B
        return 1 - probs['b']
    elif learned_gate == 'implied_by':  # A ⇐ B = 1 - B + A*B
        return 1 - probs['b'] + probs['a'] * probs['b']
    elif learned_gate == 'not_a':  # ¬A = 1 - A
        return 1 - probs['a']
    elif learned_gate == 'implies':  # A ⇒ B = 1 - A + A*B
        return 1 - probs['a'] + probs['a'] * probs['b']
    elif learned_gate == 'not_and':  # ¬(A ∧ B) = 1 - A*B
        return 1 - probs['a'] * probs['b']
    elif learned_gate == 'one':  # True
        return 1

'''
########################################################
# Hook function to track activations and logic gates for each neuron
def sf_hook(module, input, output, layer_idx):

    # Initialize activation counts and gates for the layer if not done already
    if len(activation_counts_per_layer) <= layer_idx:
        activation_counts_per_layer.append(np.zeros(output.size(1)))
        gates_per_layer.append([])
        gates_initialized.append(False)  # Mark that gates are not initialized yet for this layer

    # Store activations (output > 0 means it fired)
    activation_counts_per_layer[layer_idx] += (output > 0).sum(dim=0).cpu().numpy()

    # Only store learned logic gates if they haven't been recorded yet
    if not gates_initialized[layer_idx]:
        for neuron_idx in range(output.size(1)):
            # Get the learned gate by taking the argmax of the weights for the neuron
            gate_op_idx = module.weights[neuron_idx].argmax().item()
            learned_gate = ALL_OPERATIONS[gate_op_idx]

            # Get the input connections (indices) for the gate
            input_neuron_a = module.indices[0][neuron_idx].item()
            input_neuron_b = module.indices[1][neuron_idx].item()

            gates_per_layer[layer_idx].append({
                'Gate': learned_gate,
                'Inputs': (input_neuron_a, input_neuron_b)
            })

        # Mark that gates have been initialized for this layer
        gates_initialized[layer_idx] = True        

########################################################
'''

#You define the __add__, __sub__, and __mul__ methods for the class, that's how. Each method takes two objects (the operands of +/-/*) as arguments and is expected to return the result of the computation.
class LogicGraph:
    def __init__(self,model):  
        
        # Init data structures
        self.all_gates = ["zero", "and", "not_implies", "a", "not_implied_by", "b", "xor","or","not_or","not_xor","not_b","implied_by","not_a","implies","not_and","one"]
        self.a_gates   = [        "and", "not_implies", "a", "not_implied_by",      "xor","or","not_or","not_xor",        "implied_by","not_a","implies","not_and"      ]
        self.b_gates   = [        "and", "not_implies",      "not_implied_by", "b", "xor","or","not_or","not_xor","not_b","implied_by",        "implies","not_and"      ]
        self.net       = model.eval()
        self.nlayers   = len(self.net.logic_layers)
        self.g         = self.reset_graph()
        self.hooks     = []
    
    # Add signal tracking hooks
    def add_hooks(self): 
        # Register model hooks to track SP 
        for idx, layer in enumerate(self.net.logic_layers):
            hook = layer.register_forward_hook(lambda module, input, output, layer_idx=idx+1: self.sf_hook(module, input, output, layer_idx))
            self.hooks.append(hook)

    # Remove signal tracking hooks
    def remove_hooks(self):
        if len(self.hooks)>0: 
            # Remove hooks to clean up
            for hook in self.hooks:
                hook.remove()
        self.hooks = []
    
    # hook function to track the switching frequency 
    def sf_hook(self, module, input, output, layer_idx): 
        output = torch.sum(output, axis=0)
        for i,val in enumerate(output): 
            self.count_dict[f"L{layer_idx}_N{i}"] += val.item()
    
    # Reset the graph structrue
    def reset_graph(self): 
        
        # Reset the graph
        self.g = nx.DiGraph()
        self.g.add_node("LOW", layer=0, counts=0, gate="input", sf=0,sp=0)
        self.g.add_node("HIGH", layer=0, counts=0, gate="input", sf=0,sp=1)
        
        # Add primary layers
        for i, layer in enumerate(self.net.logic_layers): 
            for j, (aid, bid) in enumerate(zip(layer.indices[0], layer.indices[1])): 
                
                # Initialize the input names 
                if i == 0 and j == 0: 
                    self.count_dict = {f"L{i}_N{nid}":0 for nid in range(self.net.logic_layers[0].in_dim)}
                
                # Define nodes to be added
                naid = f"L{i}_N{aid}" 
                nbid = f"L{i}_N{bid}" 
                ngid = f"L{i+1}_N{j}" 
                self.count_dict[ngid] = 0  
                
                # Get learned gate type
                gate_op_idx  = layer.weights[j].argmax().item()
                learned_gate = ALL_OPERATIONS[gate_op_idx]                
                
                # Add gates if they're not included
                self.g.add_node(naid, layer=i,   counts=0, gate="input",      sf=0, sp=0) if naid not in self.g.nodes else None
                self.g.add_node(nbid, layer=i,   counts=0, gate="input",      sf=0, sp=0) if nbid not in self.g.nodes else None
                self.g.add_node(ngid, layer=i+1, counts=0, gate=learned_gate, sf=0, sp=0) if ngid not in self.g.nodes else None
                
                # Add edges if they're not connected
                if learned_gate in self.a_gates: 
                    self.g.add_edge(naid, ngid, ab="a")
                if learned_gate in self.b_gates: 
                    self.g.add_edge(nbid, ngid, ab="b") if (nbid, ngid) not in self.g.edges() else self.g.add_edge(nbid, ngid, ab="ab")
                if learned_gate == "zero": 
                    self.g.add_edge("LOW", ngid)
                if learned_gate == "one": 
                    self.g.add_edge("HIGH", ngid)
                    
                # Connect output nodes if needed
                if i == self.nlayers-1: 
                    
                    # Define nodes to be added
                    noid = f"L{self.nlayers+1}_N{int(j*self.net.group.k/(len(layer.indices[0])))}" 
                    self.count_dict[noid] = 0

                    # Add output node 
                    self.g.add_node(noid, layer=self.nlayers+1, counts=0, gate="output",sf=0, sp=0)  if noid not in self.g.nodes else None

                    # Add gate to output 
                    self.g.add_edge(ngid, noid,ab=None)
        
        return self.g 

    # Return the latent predictions for a 
    #def get_latent(self,x):
    #    x = self.net.flatten(x.to('cuda'))
    #    xlist = x.tolist()[0]
    #    for i, layer in enumerate(self.net.logic_layers): 
    #        x = layer(x)
    #        xlist = xlist + x.tolist()[0]
    #    xlist = xlist + self.net.group(x).tolist()[0]
    #    return xlist 
    
    # Add tracking functions
    def compute_sf(self, dataset): 
        
        # Add tracking functions
        #print("adding hooks")
        self.remove_hooks()
        self.add_hooks()

        # Pass all samples thorugh network with hook functions added to track properties
        #print("Processing")
        total_images = 0 
        with torch.no_grad():  # Disable gradient computation
            if isinstance(dataset,torch.utils.data.dataloader.DataLoader): 
                for batch_inputs, _ in tqdm(dataset, desc="Processing images"):
                    batch_inputs = batch_inputs.to('cuda')
                    self.net(batch_inputs.float())  # Forward pass
                    total_images += batch_inputs.size(0)
            if isinstance(dataset,torch.Tensor): 
                batch_inputs = dataset.to('cuda')
                self.net(batch_inputs)  # Forward pass
                total_images += batch_inputs.size(0)
        
        # Need a way to garuntee that all layer have some properties associated with them (initialize) 
        
        # Compute and set the switching frequency property 
        sf = {}
        for name in self.count_dict.keys(): 
            sf[name] = self.count_dict[name]/total_images
            
        nx.set_node_attributes(self.g,sf,"sf")
            
        # Remove tracking functions
        self.remove_hooks()
            
        return None

    # Add tracking functions
    def compute_sp(self, image): 
        
        # Flatten the image
        for i,p in enumerate(image.flatten()): 
            node = f"L0_N{i}"
            self.g.nodes()[node]["sp"] = p
        
        # Calcualte the sp per layer 
        for i,_ in enumerate(self.net.logic_layers): 
            nodes = [node for node in self.g.nodes if node.split("_")[0] == f"L{i+1}"]
            for node in nodes: 
                preds = list(self.g.predecessors(node))
                gate_type = self.g.nodes()[node]["gate"]
                
                # Only add the SP 
                sps = {} 
                if gate_type != "one" and gate_type != "zero": 
                    for pred in preds: 
                        sps[self.g.edges()[(pred, node)]["ab"]] = self.g.nodes()[pred]["sp"]
                
                 # Only add the SP 
                if "ab" in sps.keys():
                    sps["a"] = sps["b"] = sps["ab"] 
                
                self.g.nodes()[node]["sp"] = calculate_prob_z(gate_type, sps)
                        
    def plot_sf_hist(self,bins=100): 

        # Get node properties 
        l1 = nx.get_node_attributes(self.g,"sf")
        sf = [l1[name] for name in list(l1)]
        plt.figure()
        plt.hist(sf,bins=bins)
        plt.show()
        
        # Get node properties 
        df     = pd.DataFrame([nx.get_node_attributes(self.g,"gate"), nx.get_node_attributes(self.g,"sf"), nx.get_node_attributes(self.g,"layer")],).T
        df     = df.rename(columns={0: 'gates', 1: 'sf',2:"layer"})
        layers = np.unique(df["layer"])

        # Plot the distribution across all gate types
        for gate in ALL_OPERATIONS: 
            plt.figure(figsize=(2,2))
            for layer in layers: 
                data = df[df["gates"] == gate]
                data = data[data["layer"] == layer]
                data = data.fillna(0.0)
                plt.hist(data["sf"],bins=bins,alpha=0.5)
                plt.title(gate)
            plt.legend(layers)
            plt.show()

        # Plot the distribution across all gate types
        plt.figure(figsize=(2,2))
        for layer in layers: 
            data = df[df["layer"] == layer]
            data = data.fillna(0.0)
            plt.hist(data["sf"],bins=bins,alpha=0.5)
            plt.title(layer)
        plt.legend(layers)
        plt.show()    

        return None
    
    def forward(self, x): 
        return self.net.forward(x)

    def nodes(self):
        return self.g.nodes()

    def edges(self):
        return self.g.edges()

    def __add__(self, other): 
        return  None

    def subtract(self,other,metric): 
        new  = copy.deepcopy(self)
        sf1  = nx.get_node_attributes(new.g,metric)
        sf2  = nx.get_node_attributes(other.g,metric)
        diff = {name:sf1[name]-sf2[name] for name in sf1.keys()}
        nx.set_node_attributes(new.g,diff,metric)
        return new 

    def sub_sp(self, other): 
        return self.subtract(other,"sp")

    def __sub__(self, other): 
        return self.subtract(other,"sf")

    def find_fan_in(self, output_node, threshold, metric="sf"):
        """
        Find all input nodes connected to the given output node.
        """
        fan_in = set()
        nodes_to_check = [(output_node,1)]
        input_set = []
        connections = []
        while nodes_to_check:

            # Get node and node properties 
            parent_node, dpo = nodes_to_check.pop(0)
            new_fan_in  = []
            child_gates = []
            parent_type = self.g.nodes[parent_node]["gate"]

            # Account for all predesesors
            for child_node in list(self.g.predecessors(parent_node)): 

                desired_parent_output = dpo*1
                connections.append((child_node,parent_node))
                
                # Get type of child node 
                child_type = self.g.nodes[child_node]["gate"]

                # Account for output case 
                if parent_type == "output": 

                    # Check whether the current node is contributing to the class output
                    desired_child_output = desired_parent_output
                    if self.g.nodes[child_node][metric] > threshold: 
                        child_gates.append((child_node, desired_child_output))
                        new_fan_in.append((child_node,  desired_child_output))
                        
                # Account for hidden-layer case 
                elif child_node != "input": 

                    # Get the current wire type
                    props     = self.g.get_edge_data(child_node, parent_node)
                    wire_type = props["ab"] if "ab" in props.keys() else None

                    # Check whether the current wire increases the class score by being zero
                    if (wire_type == "a" and parent_type in a_not_list) or (wire_type == "b" and parent_type in b_not_list): 

                        # Adjust the "desired child output"
                        desired_child_output = -desired_parent_output

                        # Check whether the current node is contributing to the class output
                        if self.g.nodes[child_node][metric] <= threshold: # Less than because we contribute by being a zero 
                            child_gates.append((child_node,desired_child_output))
                            new_fan_in.append((child_node,desired_child_output))

                    # Case where the current wire increases the class score by being a one
                    else: 

                        # Adjust the "desired child output"
                        desired_child_output = desired_parent_output

                        # Check whether the current node is contributing to the class output
                        if self.g.nodes[child_node][metric] > threshold: 
                            child_gates.append((child_node,desired_child_output))
                            new_fan_in.append((child_node,desired_child_output))

                # Account for input case
                else: 

                    # Get the current wire type
                    props = self.g.get_edge_data(child_node, parent_node)
                    wire_type = props["ab"] if "ab" in props.keys() else None

                    # Check whether the current wire increases the class score by being zero
                    if (wire_type == "a" and parent_type in a_not_list) or (wire_type == "b" and parent_type in b_not_list): 

                        # Adjust the "desired child output"
                        child_sign = -1*parent_sign
                        new_fan_in.append(  (child_node, child_sign))
                        input_set.append( (child_node, child_sign))

                    # Case where the current wire increases the class score by being a one
                    else: 

                        # Adjust the "desired child output"
                        child_sign = parent_sign
                        new_fan_in.append(  (child_node, child_sign))
                        input_set.append( (child_node, child_sign))

            fan_in.update(new_fan_in)
            nodes_to_check.extend(child_gates)

        return fan_in, input_set, connections
    

    def node_to_pixel(self,node_id):
        """
        Convert a node ID (e.g., 'L0_Gate 1') to pixel coordinates.
        Assumes the input layer corresponds to a 28x28 image (MNIST format).
        """
        if (node_id != "HIGH") and (node_id != "LOW"): 
            layer, gate = node_id.split('_')

            gate_num = int(gate.split("N")[1]) - 1  # Convert 'Gate X' to zero-based index

            if layer == 'L0':  # Input layer
                y = gate_num // 20
                x = gate_num % 20
                return (x, y)

        return None  # For non-input layers, return None

    def visualize_fan_in(self, output_node,threshold,signed=True, show=False,metric="sf"):
        """
        Visualize the fan-in of the given output node as highlighted pixels in an image.
        """
        fan_in,pixels = find_fan_in(self.g, output_node,threshold,metric)
        # Create a blank 28x28 image
        img = np.zeros((20, 20))

        # Highlight pixels corresponding to input nodes in the fan-in
        for node,sign in fan_in:
            pixel = node_to_pixel(node)
            if pixel:
                x, y = pixel
                if signed: 
                    img[y, x] += sign  # Set pixel to white (1)
                else: 
                    img[y, x] += 1  # Set pixel to white (1)

        if show: 
            # Plot the image
            plt.imshow(img)
            plt.title(f"Fan-in visualization for {output_node}")
            plt.colorbar()
            plt.axis('off')
            plt.show()

        return img, fan_in,pixels    