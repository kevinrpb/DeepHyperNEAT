'''
Functions for drawing substrates and CPPNs using graphviz.

Largely copied from neat-python. (Copyright 2015-2017, CodeReclaimers, LLC.)
'''

import graphviz
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
try:
   import cPickle as pickle
except:
   import pickle

def draw_net(net, filename=None):
    '''
    Draw a network.

    net      -- the network to be drawn
    filename -- name of image file to be rendered
    '''
    # Dictionaries for node names and node colors
    node_names, node_colors = {}, {}

    # Dictionary of node attributes for graphviz
    node_attrs = {
        'shape': 'circle',
        'fontsize': '7',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph('svg', node_attr=node_attrs)

    # Set of input nodes from net
    inputs = set()
    # Traverse nodes from net and create them in dot
    for k in net.input_nodes:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled',
                       'shape': 'box',
                       'fillcolor': node_colors.get(k, 'lightgray')}
        dot.node(name, _attributes=input_attrs)
    try:
      # Set of bias nodes from net
      bias = set()
      # Traverse bias nodes and create them in visualization (if any)
      for k in net.bias_node:
          bias.add(k)
          name = node_names.get(k, str(k))
          bias_attrs = {'style': 'filled',
                         'shape': 'circle',
                         'fillcolor': node_colors.get(k, 'lightgray')}
          dot.node(name, _attributes=bias_attrs)
    except:
      pass

    # Set of output nodes from net
    outputs = set()
    # Traverse nodes from net and create them in dot
    for k in net.output_nodes:
        outputs.add(k)
        name = node_names.get(k, str(k))
        try:
            tuple_string = str(net.output_nodes[k][0])+str(net.output_nodes[k][1])
            node_attrs = {'style': 'filled',
                          'label': str(k)+"\n"+tuple_string,
                          'fillcolor': node_colors.get(k, 'lightblue'),
                          # 'shape': 'square',
                          'fontsize':'7',
                          'height': '0.45',
                          'width': '0.45',
                          'fixedsize': 'true'}
        except:
            node_attrs = {'style': 'filled',
                          # 'xlabel': str(net.output_nodes[k]),
                          'fillcolor': node_colors.get(k, 'lightblue')}
        dot.node(name, _attributes=node_attrs)

    for node, act_func, agg_func, links in net.node_evals:
        for i, w in links:
            input, output = node, int(i)
            a = node_names.get(output, str(output))
            b = node_names.get(input, str(input))
            style = 'solid'
            if w > 0.0:
                color = 'green'
            elif w == 0.0:
                color = 'purple'
            else:
                color = 'red'
            width = str(0.1 + abs(w / 5.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename)

    return dot

def plot_series(*data, labels=None, settings={}):
    # Get default settings and update with those received
    _settings = {
        'title': None,
        'ylim': None,
        'xlabel': 'x',
        'ylabel': 'y',
        'dashes': False,
        'markers': False,
        'filename': None
    }
    _settings.update(settings)

    # Make data a list and flatten if only one series
    data = list(zip(*data))

    # If no labels, make series_N the column names
    if labels is None:
        labels = list(map(lambda i: f'series_{i}', range(0, len(data))))
    # Allow passing just one string for one series
    elif isinstance(labels, str):
        labels = [labels]

    # Make the dataframe
    df = pd.DataFrame(data, columns=labels)

    # Create plot
    sns.lineplot(data=df, dashes=_settings['dashes'], markers=_settings['markers'])

    # Update title and axes
    plt.title(_settings['title'])
    plt.ylabel(_settings['ylabel'])
    plt.xlabel(_settings['xlabel'])

    plt.ylim(_settings['ylim'])

    plt.tight_layout()

    # Save the figure and flush to ensure next figure is 'reset'
    if _settings['filename'] is not None:
        plt.savefig(_settings['filename'])

    plt.clf()
