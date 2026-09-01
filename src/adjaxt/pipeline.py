from adjaxt.layers import *
from adjaxt.config import *

def init_layer(key, conf):
    if conf.init_fn is None:
        raise TypeError("Object passed to init_layer was not a valid config, or lacks init_fn field")
    return conf.init_fn(key, conf)

def flatten_dl(d, parent_key=''):
    if isinstance(d, dict):
        for key, value in d.items():
            combined_key = f"{parent_key}.{key}" if parent_key else key
            yield from flatten_dl(value, combined_key)
    elif isinstance(d, list):
        if parent_key == '':
            raise ValueError("Weight map cannot be a list, it has to be a dict (on top level)")
        for i, value in enumerate(d):
            combined_key = f"{parent_key}.{str(i)}"
            yield from flatten_dl(value, combined_key)
    else:
        if parent_key == '':
            raise ValueError("Weight map has to be a dict (on top level)")
        yield parent_key, d

def pipeline(**kwargs):
    weights = { k: init_layer(v) for k, v in kwargs.items() }
    fns = { k: v.exec_fn for k, v in kwargs.items() }
    
    @jax.jit
    def pipe(x: jax.Array, w: dict, cfg: dict):
        for k, v in kwargs.items():
            x = fns[k](x, w[k], cfg[k]) 
        return x
        
    return pipe, weights